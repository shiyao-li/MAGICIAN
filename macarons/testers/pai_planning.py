import os
import sys

import gc
import shutil
from ..utility.macarons_utils import *
from ..utility.utils import count_parameters
from ..utility.pai_utils import CamerasWrapper
import json
import time
# from ..utility.diffusion_utils import *

import pickle
import lmdb
def cleanup_trajectory_folders(training_frames_path, keep_folders=['imgs']):
    """
    Delete all subfolders except the ones specified in keep_folders.
    Only keeps the imgs folder and deletes frames, depths, occupancy folders.

    Args:
        training_frames_path: Path to the frames folder (e.g., .../training/0/frames/)
        keep_folders: List of folder names to keep (default: ['imgs'])
    """
    # Get parent directory (e.g., .../training/0/)
    trajectory_dir = os.path.dirname(training_frames_path)

    if not os.path.exists(trajectory_dir):
        print(f"Warning: Trajectory directory does not exist: {trajectory_dir}")
        return

    deleted_folders = []
    kept_folders = []

    print(f"\n=== Cleaning up trajectory directory: {trajectory_dir} ===")

    # Iterate through all items in the trajectory directory
    for item in os.listdir(trajectory_dir):
        item_path = os.path.join(trajectory_dir, item)

        # Only process directories
        if os.path.isdir(item_path):
            if item in keep_folders:
                kept_folders.append(item)
                print(f"  ✓ Keeping folder: {item}")
            else:
                # Delete the folder and all its contents
                shutil.rmtree(item_path)
                deleted_folders.append(item)
                print(f"  ✗ Deleted folder: {item}")

    print(f"\nCleanup summary:")
    print(f"  - Kept: {kept_folders}")
    print(f"  - Deleted: {deleted_folders}\n")

def save_to_lmdb(lmdb_env, key, data_dict):
    """
    Save data to LMDB database.

    Args:
        lmdb_env: LMDB environment
        key: string key (e.g., "scene_name/start_cam_idx")
        data_dict: dictionary containing the data to save
    """
    with lmdb_env.begin(write=True) as txn:
        # Serialize data using pickle
        serialized_data = pickle.dumps(data_dict)
        txn.put(key.encode('utf-8'), serialized_data)
    print(f"Saved data to LMDB with key: {key}")

def load_from_lmdb(lmdb_env, key):
    """
    Load data from LMDB database.

    Args:
        lmdb_env: LMDB environment
        key: string key (e.g., "scene_name/start_cam_idx")

    Returns:
        data_dict: deserialized dictionary
    """
    with lmdb_env.begin() as txn:
        serialized_data = txn.get(key.encode('utf-8'))
        if serialized_data is None:
            return None
        return pickle.loads(serialized_data)


# ==================== RaDe-GS Integration ====================
RADE_GS_PATH = os.path.join(os.path.dirname(__file__), "../../RaDe-GS")
if RADE_GS_PATH not in sys.path:
    sys.path.insert(0, RADE_GS_PATH)


class SimpleGaussianModel:
    def __init__(self, means, opacities, scales, rotations, colors, device):
        """
        Args:
            means: (N, 3) locations
            opacities: (N, 1) opacity[0, 1]
            scales: (N, 3) 
            rotations: (N, 4) 
            colors: (N, 3) 
        """
        self.device = device
        self._xyz = means.to(device)
        self._opacity = self.inverse_sigmoid(opacities.to(device))  # logit
        self._scaling = torch.log(scales.to(device))  # log
        self._rotation = rotations.to(device)
        self._colors_precomp = colors.to(device)  

        self.active_sh_degree = 0
        self.max_sh_degree = 0
        self.max_radii2D = torch.zeros(means.shape[0], device=device)

    @staticmethod
    def inverse_sigmoid(x, eps=1e-6):
        """ logit: logit(x) = log(x / (1-x))"""
        x = torch.clamp(x, eps, 1 - eps)
        return torch.log(x / (1 - x))

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        # none
        return torch.zeros(self._xyz.shape[0], 1, 3, device=self.device)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def get_opacity_with_3D_filter(self):
        return self.get_opacity

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_scaling_n_opacity_with_3D_filter(self):
        return self.get_scaling, self.get_opacity

    @property
    def get_colors_precomp(self):
        return self._colors_precomp


def render_gaussian_depth(gaussian_means, gaussian_opacities, gaussian_scales,
                          gaussian_rotations, gaussian_colors, gs_camera, device,
                          bg_color=None, kernel_size=0.1):
    """

    Args:
        gaussian_means: (N, 3) 
        gaussian_opacities: (N, 1)
        gaussian_scales: (N, 3) 
        gaussian_rotations: (N, 4) 
        gaussian_colors: (N, 3) 
        gs_camera: GSCamera 
        device: torch device
        bg_color: 
        kernel_size: Mip-Splatting kernel size

    Returns:
        rendered_depth: (1, H, W) median depth
        rendered_image: (3, H, W) RGB image
    """
    if bg_color is None:
        bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)

    gaussians = SimpleGaussianModel(
        means=gaussian_means,
        opacities=gaussian_opacities,
        scales=gaussian_scales,
        rotations=gaussian_rotations,
        colors=gaussian_colors,
        device=device
    )
    import math
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    tanfovx = math.tan(gs_camera.FoVx * 0.5)
    tanfovy = math.tan(gs_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(gs_camera.image_height),
        image_width=int(gs_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=gs_camera.world_view_transform,
        projmatrix=gs_camera.full_proj_transform,
        sh_degree=0,
        campos=gs_camera.camera_center,
        prefiltered=False,
        require_coord=False,
        require_depth=True,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussians.get_xyz
    means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=False, device=device)
    scales, opacity = gaussians.get_scaling_n_opacity_with_3D_filter
    rotations = gaussians.get_rotation
    colors_precomp = gaussians.get_colors_precomp

    with torch.no_grad():
        rendered_image, radii, _, _, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )

    return rendered_median_depth, rendered_image


def update_gaussian_colors_from_knowledge(knowledge_values):
    """
    use knowledge_values update Gaussian colors

    Args:
        knowledge_values: 
            0 = unknown → white [1,1,1]
            1 = known → black [0,0,0]
    """
    inverted_values = 1.0 - knowledge_values
    colors = inverted_values.unsqueeze(1).repeat(1, 3)
    return colors

# ==================== End RaDe-GS Integration ====================

def load_current_frame_perfect_depth(camera, device):
    current_frame_nb = camera.n_frames_captured - 1
    frame_path = os.path.join(camera.save_dir_path, str(current_frame_nb) + '.pt')

    
    frame_dict = torch.load(frame_path, map_location=device)
    
    return {
        'rgb': frame_dict['rgb'],           # (1, H, W, 3)
        'zbuf': frame_dict['zbuf'],         # (1, H, W, 1) 
        'mask': frame_dict['mask'],         # (1, H, W, 1)
        'R': frame_dict['R'],               # (1, 3, 3)
        'T': frame_dict['T'],               # (1, 3)
        'zfar': camera.zfar
    }

def apply_perfect_depth_simple(frame_data, device, use_error_mask=True):
    images = frame_data['rgb']
    zbuf = frame_data['zbuf'] 
    mask = frame_data['mask'].bool()
    R = frame_data['R']
    T = frame_data['T']
    
    # GT zbuf
    depth = torch.clamp(zbuf, min=0.5, max=750.0) 
    
    if use_error_mask:
        error_mask = mask
    else:
        error_mask = torch.ones_like(mask)
    
    return depth, mask, error_mask, R, T


def clear_folder(folder_path):
    """
    remove all for a folder
    """
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print(f" {folder_path} is clear")

dir_path = os.path.abspath(os.path.dirname(__file__))
# data_path = os.path.join(dir_path, "../../../../../../datasets/rgb")
data_path = os.path.join(dir_path, "../../data/scenes")
results_dir = os.path.join(dir_path, "../../results/scene_exploration")
weights_dir = os.path.join(dir_path, "../../weights/macarons")
configs_dir = os.path.join(dir_path, "../../configs/macarons")

def setup_test(params, model_path, device, verbose=True):
    # Create dataloader
    _, _, test_dataloader = get_dataloader(train_scenes=params.train_scenes,
                                           val_scenes=params.val_scenes,
                                           test_scenes=params.test_scenes,
                                           batch_size=1,
                                           ddp=False, jz=False,
                                           world_size=None, ddp_rank=None,
                                           data_path=params.data_path)
    print("\nThe following scenes will be used to test the model:")
    for batch, elem in enumerate(test_dataloader):
        print(elem['scene_name'][0])

    # Create model
    macarons = load_pretrained_macarons(pretrained_model_path=params.pretrained_model_path,
                                        device=device, learn_pose=params.learn_pose)


    trained_weights = torch.load(model_path, map_location=device, weights_only=False)
    macarons.load_state_dict(trained_weights["model_state_dict"], ddp=True)  # todo: replace by params.ddp
    depth_losses = np.array(trained_weights["depth_losses"])
    depth_losses_per_epoch = (depth_losses[::2] + depth_losses[1::2]) / 2
    # depth_losses_per_epoch = depth_losses
    print("\nModel name:", model_path)
    print("\nThe model has", (count_parameters(macarons.depth) + count_parameters(macarons.scone)) / 1e6,
          "trainable parameters.")
    print("It has been trained for", trained_weights["epoch"], "epochs.")
    print("The loss was:", depth_losses_per_epoch[-1], depth_losses_per_epoch[-1] * 3 / 4)
    print(params.n_alpha, "additional frames are used for depth prediction.")

    # Set loss functions
    pose_loss_fn = get_pose_loss_fn(params)
    regularity_loss_fn = get_regularity_loss_fn(params)
    ssim_loss_fn = None
    if params.training_mode == 'self_supervised':
        depth_loss_fn = get_reconstruction_loss_fn(params)
        ssim_loss_fn = get_ssim_loss_fn(params)
    else:
        raise NameError("Invalid training mode.")
    occ_loss_fn = get_occ_loss_fn(params)
    cov_loss_fn = get_cov_loss_fn(params)

    # Creating memory
    print("\nUsing memory folders", params.memory_dir_name)
    scene_memory_paths = []
    for scene_name in params.test_scenes:
        scene_path = os.path.join(test_dataloader.dataset.data_path, scene_name)
        scene_memory_path = os.path.join(scene_path, params.memory_dir_name)
        scene_memory_paths.append(scene_memory_path)
    memory = Memory(scene_memory_paths=scene_memory_paths, n_trajectories=params.n_memory_trajectories,
                    current_epoch=0, verbose=verbose)

    return test_dataloader, macarons, memory


def setup_test_scene(params,
                     mesh,
                     settings,
                     mirrored_scene,
                     device,
                     mirrored_axis=None,
                     surface_scene_feature_dim=1,
                     test_resolution=0.05,
                     covered_scene_feature_dim=1):
    """
    Setup the different scene objects used for prediction and performance evaluation.

    :param params:
    :param mesh:
    :param settings:
    :param device:
    :param is_master:
    :return:
    """

    # Initialize gt_scene: we use this scene to store gt surface points to evaluate the performance of the model.
    # This scene is not used for supervision during training, since the model is self-supervised from RGB data
    # captured in real-time.
    gt_scene = Scene(x_min=settings.scene.x_min,
                     x_max=settings.scene.x_max,
                     grid_l=settings.scene.grid_l,
                     grid_w=settings.scene.grid_w,
                     grid_h=settings.scene.grid_h,
                     cell_capacity=params.surface_cell_capacity,
                     cell_resolution=test_resolution * params.scene_scale_factor,
                     n_proxy_points=params.n_proxy_points,
                     device=device,
                     view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                     feature_dim=3,
                     mirrored_scene=mirrored_scene,
                     mirrored_axis=mirrored_axis)  # We use colors as features

    covered_scene = Scene(x_min=settings.scene.x_min,
                          x_max=settings.scene.x_max,
                          grid_l=settings.scene.grid_l,
                          grid_w=settings.scene.grid_w,
                          grid_h=settings.scene.grid_h,
                          cell_capacity=params.surface_cell_capacity,
                          cell_resolution=test_resolution * params.scene_scale_factor,
                          n_proxy_points=params.n_proxy_points,
                          device=device,
                          view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                          feature_dim=covered_scene_feature_dim,
                          mirrored_scene=mirrored_scene,
                          mirrored_axis=mirrored_axis)  # We use colors as features

    # We fill gt_scene with points sampled on the surface of the ground truth mesh
    gt_surface, gt_normals, gt_surface_colors = get_scene_gt_surface(gt_scene=gt_scene,
                                                         verts=mesh.verts_list()[0],
                                                         faces=mesh.faces_list()[0],
                                                         n_surface_points=params.n_gt_surface_points,
                                                         return_colors=True,
                                                         mesh=mesh)
    gt_scene.fill_cells(gt_surface, features=gt_surface_colors)

    # Initialize surface_scene: we store in this scene the surface points computed by the depth model from RGB images
    surface_scene = Scene(x_min=settings.scene.x_min,
                          x_max=settings.scene.x_max,
                          grid_l=settings.scene.grid_l,
                          grid_w=settings.scene.grid_w,
                          grid_h=settings.scene.grid_h,
                          cell_capacity=params.surface_cell_capacity,
                          cell_resolution=None,
                          n_proxy_points=params.n_proxy_points,
                          device=device,
                          view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                          feature_dim=surface_scene_feature_dim,  # We use visibility history as features
                          mirrored_scene=mirrored_scene,
                          mirrored_axis=mirrored_axis)

    # Initialize proxy_scene: we store in this scene the proxy points
    proxy_scene = Scene(x_min=settings.scene.x_min,
                        x_max=settings.scene.x_max,
                        grid_l=settings.scene.grid_l,
                        grid_w=settings.scene.grid_w,
                        grid_h=settings.scene.grid_h,
                        cell_capacity=params.proxy_cell_capacity,
                        cell_resolution=params.proxy_cell_resolution,
                        n_proxy_points=params.n_proxy_points,
                        device=device,
                        view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                        feature_dim=1,  # We use proxy points indices as features
                        mirrored_scene=mirrored_scene,
                        score_threshold=params.score_threshold,
                        mirrored_axis=mirrored_axis)
    proxy_scene.initialize_proxy_points()

    return gt_scene, covered_scene, surface_scene, proxy_scene


def setup_test_camera(params,
                      mesh, intersector, start_cam_idx,
                      settings,
                      occupied_pose_data,
                      device,
                      training_frames_path,
                      mirrored_scene=False,
                      mirrored_axis=None):
    """
    Setup the camera used for prediction.

    :param params:
    :param mesh:
    :param start_cam_idx:
    :param settings:
    :param occupied_pose_data:
    :param device:
    :param training_frames_path:
    :return:
    """
    # Default camera to initialize the renderer
    n_camera = 1
    camera_dist = [10 * params.scene_scale_factor] * n_camera  # 10
    camera_elev = [30] * n_camera
    camera_azim = [260] * n_camera  # 160
    R, T = look_at_view_transform(camera_dist, camera_elev, camera_azim)
    zfar = params.zfar
    fov_camera = FoVPerspectiveCameras(R=R, T=T, zfar=zfar, device=device)

    renderer = get_rgb_renderer(image_height=params.image_height,
                                image_width=params.image_width,
                                ambient_light_intensity=params.ambient_light_intensity,
                                cameras=fov_camera,
                                device=device,
                                max_faces_per_bin=200000
                                )

    # Initialize camera
    camera = Camera(x_min=settings.camera.x_min, x_max=settings.camera.x_max,
                    pose_l=settings.camera.pose_l, pose_w=settings.camera.pose_w, pose_h=settings.camera.pose_h,
                    pose_n_elev=settings.camera.pose_n_elev, pose_n_azim=settings.camera.pose_n_azim,
                    n_interpolation_steps=params.n_interpolation_steps, zfar=params.zfar,
                    renderer=renderer,
                    device=device,
                    contrast_factor=settings.camera.contrast_factor,
                    gathering_factor=params.gathering_factor,
                    occupied_pose_data=occupied_pose_data,
                    save_dir_path=training_frames_path,
                    mirrored_scene=mirrored_scene,
                    mirrored_axis=mirrored_axis)  # Change or remove this path during inference or test

    # Move to a valid neighbor pose before starting training.
    # Thus, we will have a few images to start training the depth module
    start_pose, _ = camera.get_pose_from_idx(start_cam_idx)
    # X_neighbor, V_neighbor, fov_neighbor = camera.get_camera_parameters_from_pose(start_pose)
    # current_origin = X_neighbor[0].cpu().numpy()

    # neighbor_indices = camera.get_neighboring_poses(pose_idx=start_cam_idx)
    # valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)
    # # first_cam_idx = valid_neighbors[np.random.randint(low=0, high=len(valid_neighbors))]
    # for neighbor_i in range(len(valid_neighbors)):
    #     neighbor_idx = valid_neighbors[neighbor_i]
    #     neighbor_pose, _ = camera.get_pose_from_idx(neighbor_idx)
    #     X_neighbor, V_neighbor, fov_neighbor = camera.get_camera_parameters_from_pose(neighbor_pose)
    #     ray_origin = X_neighbor[0].cpu().numpy()

    #     if not line_segment_mesh_intersection(current_origin, ray_origin, intersector):
    #         first_cam_idx = neighbor_idx
    #         break
        

    # Select a random, valid camera pose as starting pose
    camera.initialize_camera(start_cam_idx=start_cam_idx)

    # Capture initial image
    camera.capture_image(mesh)

    # We capture images along the way
    # interpolation_step = 1
    # for i in range(camera.n_interpolation_steps):
    #     camera.update_camera(start_cam_idx, interpolation_step=interpolation_step)
    #     camera.capture_image(mesh)
    #     interpolation_step += 1

    return camera

from macarons.utility.tsp_utils import generate_key_value_splited_dict, line_segment_mesh_intersection

def compute_pai_trajectory(params, macarons, camera, gt_scene, surface_scene,
                           proxy_scene, covered_scene, mesh, intersector, device, settings,
                           test_resolution=0.05, use_perfect_depth_map=False,
                           compute_collision=False):

    macarons.eval()

    # compute scene_scales
    scene_bbox_x = settings.scene.x_max[0] - settings.scene.x_min[0]
    scene_bbox_y = settings.scene.x_max[1] - settings.scene.x_min[1]
    scene_bbox_z = settings.scene.x_max[2] - settings.scene.x_min[2]
    scene_scale = (scene_bbox_x + scene_bbox_y + scene_bbox_z) / 3.0
    print(f"Scene scale computed: {scene_scale:.2f} (bbox: x={scene_bbox_x:.2f}, y={scene_bbox_y:.2f}, z={scene_bbox_z:.2f})")

    # curriculum_distances = get_curriculum_sampling_distances(params, surface_scene, proxy_scene)
    splited_pose_space_idx = camera.generate_new_splited_dict()
    splited_pose_space = generate_key_value_splited_dict(splited_pose_space_idx)

    full_pc = torch.zeros(0, 3, device=device)
    full_pc_colors = torch.zeros(0, 3, device=device)
    full_pc_idx = torch.zeros(0, 1, device=device)
    coverage_evolution = []
    t0 = time.time()
    pose_i = 0

    def process_current_frame():
        current_frame = load_current_frame_perfect_depth(camera, device)
        depth, mask, error_mask, R, T = apply_perfect_depth_simple(current_frame, device)
        
        fov_camera = camera.get_fov_camera_from_RT(R_cam=R, T_cam=T)
        X_cam = fov_camera.get_camera_center() 
        
        part_pc, part_pc_features = camera.compute_partial_point_cloud(
            depth=depth, mask=(mask * error_mask).bool(), images=current_frame['rgb'],
            fov_cameras=fov_camera,
            gathering_factor=params.gathering_factor * 2,
            fov_range=params.sensor_range
        )
        
        fov_proxy_points, fov_proxy_mask = camera.get_points_in_fov(
            proxy_scene.proxy_points, return_mask=True,
            fov_camera=None, fov_range=params.sensor_range
        )
        
        sgn_dists = None
        if fov_proxy_mask.any():
            sgn_dists = camera.get_signed_distance_to_depth_maps(
                pts=fov_proxy_points, depth_maps=depth,
                mask=mask, fov_camera=None
            )
        
        return {
            'part_pc': part_pc,
            'part_pc_features': part_pc_features,
            'fov_proxy_points': fov_proxy_points,
            'fov_proxy_mask': fov_proxy_mask,
            'sgn_dists': sgn_dists,
            'X_cam': X_cam,  
            'current_frame': current_frame
        }
            

    while pose_i <= params.n_poses_in_trajectory:
        if pose_i % 10 == 0:
            print("Processing pose", str(pose_i) + "...")
        
        camera.fov_camera_0 = camera.fov_camera

        if pose_i > 0 and pose_i % params.recompute_surface_every_n_loop == 0:
            print("Recomputing surface...")
            fill_surface_scene(surface_scene, full_pc,
                               random_sampling_max_size=params.n_gt_surface_points,
                               min_n_points_per_cell_fill=3,
                               progressive_fill=params.progressive_fill,
                               max_n_points_per_fill=params.max_points_per_progressive_fill)

        frame_data = process_current_frame()

        # unpdate scene
        part_pc_features = torch.zeros(len(frame_data['part_pc']), 1, device=device)
        covered_scene.fill_cells(frame_data['part_pc'], features=part_pc_features)
        surface_scene.fill_cells(frame_data['part_pc'], features=part_pc_features)
        full_pc = torch.vstack((full_pc, frame_data['part_pc']))
        full_pc_colors = torch.vstack((full_pc_colors, frame_data['part_pc_features']))
        part_pc_idx = torch.full((frame_data['part_pc'].shape[0], 1), pose_i, device=device)
        full_pc_idx = torch.vstack((full_pc_idx, part_pc_idx))

        if frame_data['fov_proxy_mask'].any():
            fov_proxy_indices = proxy_scene.get_proxy_indices_from_mask(frame_data['fov_proxy_mask'])
            proxy_scene.fill_cells(frame_data['fov_proxy_points'], 
                                 features=fov_proxy_indices.view(-1, 1))
            
            proxy_scene.update_proxy_view_states(
                camera, frame_data['fov_proxy_mask'],
                signed_distances=frame_data['sgn_dists'],
                distance_to_surface=None, 
                X_cam=frame_data['X_cam']  
            )
            
            proxy_scene.update_proxy_supervision_occ(
                frame_data['fov_proxy_mask'], frame_data['sgn_dists'], 
                tol=params.carving_tolerance
            )
            proxy_scene.update_proxy_out_of_field(frame_data['fov_proxy_mask'])

        surface_scene.set_all_features_to_value(value=1.)

        # compute coverage gain
        current_coverage = gt_scene.scene_coverage(
            covered_scene, surface_epsilon=2 * test_resolution * params.scene_scale_factor
        )
        if pose_i % 5 == 0:
            print("==========current coverage:", current_coverage)
        current_cov = current_coverage[0].item() if current_coverage[0] != 0. else 0.
        coverage_evolution.append(current_cov / settings.scene.visibility_ratio)

        # if pose_i >= params.n_poses_in_trajectory:
        #     break

        with torch.no_grad():
            X_world, view_harmonics, occ_probs = compute_scene_occupancy_probability_field(
                params, macarons.scone, camera, surface_scene, proxy_scene, device
            )
        filtered_X_world = X_world[occ_probs.squeeze() > 0.5]
        # pc_plot = plot_point_cloud(filtered_X_world, torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(filtered_X_world.shape[0], 1), name='Filtered reconstructed surface points',
        #             point_size=2, max_points=150000, width=800, height=600, cmap='rgb')
        # pc_plot.show()
        print(filtered_X_world.shape)
        n_points = filtered_X_world.shape[0]
        from ..utility.pai_utils import convert_camera_from_pytorch3d_to_gs
        gaussian_means = filtered_X_world  # (N, 3) 
        occ_values = occ_probs[occ_probs.squeeze() >= 0.5] 
        gaussian_opacities = occ_values    # (N, 1) 
        gaussian_scales = torch.ones(n_points, 3, device=device) * (0.7154/2)  
        gaussian_rotations = torch.tensor([[1, 0, 0, 0]], device=device, dtype=torch.float32).repeat(n_points, 1) 
        knowledge_values = torch.zeros(n_points, device=device)  # (N,)
        gaussian_colors = update_gaussian_colors_from_knowledge(knowledge_values)  # (N, 3)

        if pose_i == 0:
            sample_X_cam = camera.X_cam_history[0].view(1, 3)
            sample_V_cam = camera.V_cam_history[0].view(1, 2)
            R_sample, T_sample = get_camera_RT(sample_X_cam, sample_V_cam)
            sample_camera = FoVPerspectiveCameras(R=R_sample, T=T_sample, zfar=camera.zfar, device=device)
            K_matrix = sample_camera.get_projection_transform().get_matrix().transpose(-1, -2)

        # 1. initialize all knowledge_values to 0
        knowledge_values = torch.zeros(n_points, device=device)

        # 2. revisit all previous cameras
        history_length = len(camera.X_cam_history)

        for cam_idx in range(history_length):
            current_X_cam = camera.X_cam_history[cam_idx]
            current_V_cam = camera.V_cam_history[cam_idx]
            X_cam = current_X_cam.view(1, 3)
            V_cam = current_V_cam.view(1, 2)
            R_cam, T_cam = get_camera_RT(X_cam, V_cam)
            current_fov_camera = FoVPerspectiveCameras(R=R_cam, T=T_cam, zfar=camera.zfar, device=device)
            current_fov_camera.K = K_matrix  

            gs_cameras = convert_camera_from_pytorch3d_to_gs(
                current_fov_camera,
                height=camera.image_height,
                width=camera.image_width,
                device=device
            )
            gs_camera = gs_cameras[0]

            with torch.no_grad():
                rendered_depth, _ = render_gaussian_depth(
                    gaussian_means=gaussian_means,
                    gaussian_opacities=gaussian_opacities,
                    gaussian_scales=gaussian_scales,
                    gaussian_rotations=gaussian_rotations,
                    gaussian_colors=gaussian_colors,
                    gs_camera=gs_camera,
                    device=device,
                    bg_color=torch.tensor([1.0, 1.0, 1.0], device=device),
                    kernel_size=0.01
                )
                current_depth_map = rendered_depth[0]

                current_visible_mask = camera.check_point_visibility_from_depth(
                    filtered_X_world, current_fov_camera, current_depth_map, depth_tolerance=1.0
                )
                knowledge_values[current_visible_mask] = 1.0

        print(f"historical: {knowledge_values.sum().item()}/{n_points}")

        # 3. Beam Search 
        remaining_steps = params.n_poses_in_trajectory + 1 - history_length
        print(f"Beam search remain: {remaining_steps} steps")

        # initialize beam search
        initial_pose_idx = camera.cam_idx
        beams = [{
            'trajectory': [],
            'knowledge_values': knowledge_values.clone(),
            'score': knowledge_values.sum().item(),
            'total_coverage_gain': 0.0,  
            'current_pose_idx': initial_pose_idx
        }]
        beam_width = 10
        remaining_steps = 10
        for bs_i in range(remaining_steps):
            print(f"Beam search step {bs_i + 1}/{remaining_steps}")

            all_candidates = []

            # extend to every beams
            for beam in beams:
                neighbor_indices = camera.get_neighboring_poses(pose_idx=beam['current_pose_idx'])
                valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)

                rendering_candidate = []
                idx_candidate = []

                current_pose, _ = camera.get_pose_from_idx(beam['current_pose_idx'])
                X_current, _, _ = camera.get_camera_parameters_from_pose(current_pose)
                current_loc = X_current[0].cpu().numpy()

                for row in valid_neighbors:
                    neighbor_pose, _ = camera.get_pose_from_idx(row)
                    X_neighbor, V_neighbor, fov_neighbor = camera.get_camera_parameters_from_pose(neighbor_pose)
                    target_loc = X_neighbor[0].cpu().numpy()

                    if line_segment_mesh_intersection(current_loc, target_loc, intersector):
                        continue

                    rendering_candidate.append(fov_neighbor)
                    idx_candidate.append(row)

                if len(rendering_candidate) == 0:
                    continue

                # rendering for every pose
                for j, pose_idx in enumerate(idx_candidate):
                    fov_camera = rendering_candidate[j]
                    fov_camera.K = K_matrix

                    gs_cameras = convert_camera_from_pytorch3d_to_gs(
                        fov_camera,
                        height=camera.image_height,
                        width=camera.image_width,
                        device=device
                    )
                    gs_camera = gs_cameras[0]

                    # update colors
                    current_knowledge = beam['knowledge_values']
                    gaussian_colors = update_gaussian_colors_from_knowledge(current_knowledge)

                    with torch.no_grad():
                        rendered_depth, rendered_image = render_gaussian_depth(
                            gaussian_means=gaussian_means,
                            gaussian_opacities=gaussian_opacities,
                            gaussian_scales=gaussian_scales,
                            gaussian_rotations=gaussian_rotations,
                            gaussian_colors=gaussian_colors,
                            gs_camera=gs_camera,
                            device=device,
                            bg_color=torch.tensor([1.0, 1.0, 1.0], device=device),
                            kernel_size=0.01
                        )
                        depth_map = rendered_depth[0]

                        # compute visible mask
                        visible_mask = camera.check_point_visibility_from_depth(
                            filtered_X_world, fov_camera, depth_map, depth_tolerance=1.0
                        )

                        # white pixels: unseen points（knowledge_values=0）
                        valid_depth_mask = depth_map > 0
                        rgb_image = rendered_image  # shape: [3, H, W]
                        grayscale = rgb_image.mean(dim=0)  # [H, W]

               
                        depth_threshold = scene_scale / 2.0  

                        if valid_depth_mask.any():
                            nb_observed_pts_per_pixel = (depth_map / depth_threshold) ** 2
                            depth_weight = nb_observed_pts_per_pixel.clamp_max(1.0)

                            # compute coverage gain by using novelty map and depth weights map
                            coverage_gain = (grayscale * depth_weight * valid_depth_mask.float()).sum().item()
                        else:
                            coverage_gain = 0.0

                        new_knowledge = current_knowledge.clone()
                        new_knowledge[visible_mask] = 1.0

                        new_total_coverage_gain = beam['total_coverage_gain'] + coverage_gain

                        all_candidates.append({
                            'trajectory': beam['trajectory'] + [pose_idx],
                            'knowledge_values': new_knowledge,
                            'coverage_gain': coverage_gain,  # single step
                            'total_coverage_gain': new_total_coverage_gain,  # 路径累积增益
                            'current_pose_idx': pose_idx
                        })

            if len(all_candidates) == 0:
                print("No valid candidates found!")
                break

            # coverage gains based on rgb imgs
            all_candidates.sort(key=lambda x: x['total_coverage_gain'], reverse=True)
            beams = all_candidates[:beam_width]
            # print(f"Step {bs_i + 1}: Best total_coverage_gain = {beams[0]['total_coverage_gain']:.2f}, Score = {beams[0]['score']}/{n_points}, Current step gain = {beams[0].get('coverage_gain', 0):.2f}")
            print(f"Top {min(beam_width, len(all_candidates))} beams selected from {len(all_candidates)} candidates")

        if len(beams) > 0:
            best_beam = beams[0]
            best_trajectory = best_beam['trajectory']
        else:
            print("No valid trajectory found!")
            break

        # move one step
        next_idx = best_trajectory[0]
        print(f"move one step: pose_idx = {next_idx}")

        interpolation_step = 1
        for i in range(camera.n_interpolation_steps):
            camera.update_camera(next_idx, interpolation_step=interpolation_step)
            camera.capture_image(mesh)
            interpolation_step += 1

        pose_i += 1

    # pc_plot = plot_point_cloud(full_pc, torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(full_pc.shape[0], 1), name='Filtered reconstructed surface points', 
    #                 point_size=2, max_points=150000, width=800, height=600, cmap='rgb')
    # pc_plot.show()

    print("Trajectory computed in", time.time() - t0, "seconds.")
    print("Coverage Evolution:", coverage_evolution)
    
    return coverage_evolution, camera.X_cam_history, camera.V_cam_history, full_pc, full_pc_colors, full_pc_idx
        
def run_pai_test(params_name,
             model_name,
             results_json_name,
             numGPU,
             test_scenes,
             test_resolution=0.05,
             use_perfect_depth_map=False,
             compute_collision=False,
             load_json=False,
             dataset_path=None):

    params_path = os.path.join(configs_dir, params_name)
    weights_path = os.path.join(weights_dir, model_name)
    results_json_path = os.path.join(results_dir, results_json_name)

    params = load_params(params_path)
    params.test_scenes = test_scenes
    params.jitter_probability = 0.
    params.symmetry_probability = 0.
    params.anomaly_detection = False
    params.memory_dir_name = "test_memory_" + str(numGPU)

    params.jz = False
    params.numGPU = numGPU
    params.WORLD_SIZE = 1
    params.batch_size = 1
    params.total_batch_size = 1

    if dataset_path is None:
        params.data_path = data_path
    else:
        params.data_path = dataset_path

    # Setup device
    device = setup_device(params, None)

    # Setup model and dataloader
    dataloader, macarons, memory = setup_test(params, weights_path, device)

    # Result json
    # if load_json:
    #     with open(results_json_path, "r") as read_content:
    #         dict_to_save = json.load(read_content)
    # else:
    #     dict_to_save = {}
    lmdb_dir = os.path.join(results_dir, "ours_lmdb")
    os.makedirs(lmdb_dir, exist_ok=True)
    print(f"\nLMDB database directory: {lmdb_dir}")

    for i in range(len(dataloader.dataset)):
        scene_dict = dataloader.dataset[i]

        scene_names = [scene_dict['scene_name']]
        obj_names = [scene_dict['obj_name']]
        all_settings = [scene_dict['settings']]
        occupied_pose_datas = [scene_dict['occupied_pose']]

        batch_size = len(scene_names)

        for i_scene in range(batch_size):
            mesh = None
            torch.cuda.empty_cache()

            scene_name = scene_names[i_scene]
            obj_name = obj_names[i_scene]
            settings = all_settings[i_scene]
            settings = Settings(settings, device, params.scene_scale_factor)
            occupied_pose_data = occupied_pose_datas[i_scene]
            print("\nScene name:", scene_name)
            print("-------------------------------------")

            # dict_to_save[scene_name] = {}

            scene_path = os.path.join(dataloader.dataset.data_path, scene_name)
            mesh_path = os.path.join(scene_path, obj_name)
            # segmented_mesh_path = os.path.join(scene_path, 'segmented.obj')

            mirrored_scene = False
            mirrored_axis = None

            # Load segmented mesh
            # mesh = load_scene(segmented_mesh_path, params.scene_scale_factor, device, mirror=mirrored_scene)
            # gt_scene, _, _, _ = setup_test_scene(params,
            #                                      mesh,
            #                                      settings,
            #                                      mirrored_scene,
            #                                      device)

            # Load mesh
            mesh = load_scene(mesh_path, params.scene_scale_factor, device,
                              mirror=mirrored_scene, mirrored_axis=mirrored_axis)
            import trimesh
            mesh_for_check = trimesh.load(mesh_path)

            if isinstance(mesh_for_check, trimesh.Scene):
                # merge scenes
                mesh_for_check = mesh_for_check.dump(concatenate=True)
            mesh_for_check.vertices *= params.scene_scale_factor

            intersector = mesh_for_check.ray

            print("Mesh Vertices shape:", mesh.verts_list()[0].shape)
            print("Min Vert:", torch.min(mesh.verts_list()[0], dim=0)[0],
                  "\nMax Vert:", torch.max(mesh.verts_list()[0], dim=0)[0])

            # Use memory info to set frames and poses path
            scene_memory_path = os.path.join(scene_path, params.memory_dir_name)

            torch.cuda.empty_cache()

            for start_cam_idx_i in range(len(settings.camera.start_positions)):
                start_cam_idx = settings.camera.start_positions[start_cam_idx_i]
                print("\n" + "="*60)
                print(f"Start cam index {start_cam_idx_i} for {scene_name}: {start_cam_idx}")
                print("="*60)

                # Each start_cam_idx_i gets its own trajectory number
                trajectory_nb = start_cam_idx_i
                training_frames_path = memory.get_trajectory_frames_path(scene_memory_path, trajectory_nb)
                print(f"Using trajectory folder: {training_frames_path}")

                # Setup the Scene and Camera objects
                gt_scene, covered_scene, surface_scene, proxy_scene = None, None, None, None
                gc.collect()
                torch.cuda.empty_cache()
                gt_scene, covered_scene, surface_scene, proxy_scene = setup_test_scene(params,
                                                                                       mesh,
                                                                                       settings,
                                                                                       mirrored_scene,
                                                                                       device,
                                                                                       mirrored_axis=mirrored_axis,
                                                                                       test_resolution=test_resolution)

                # clear_folder(training_frames_path)
                camera = setup_test_camera(params, mesh, intersector, start_cam_idx, settings, occupied_pose_data,
                                           device, training_frames_path,
                                           mirrored_scene=mirrored_scene, mirrored_axis=mirrored_axis)
                print(camera.X_cam_history[0], camera.V_cam_history[0])

                coverage_evolution, X_cam_history, V_cam_history, full_pc, full_pc_colors, full_pc_idx = compute_pai_trajectory(params, macarons,
                                                                                      camera,
                                                                                      gt_scene, surface_scene,
                                                                                      proxy_scene, covered_scene,
                                                                                      mesh,
                                                                                      intersector,
                                                                                      device,
                                                                                      settings,
                                                                                      test_resolution=test_resolution,
                                                                                      use_perfect_depth_map=use_perfect_depth_map,
                                                                                      compute_collision=compute_collision)
                

                # Open LMDB, save data, then close
                print(f"\n=== Saving trajectory data to LMDB ===")
                lmdb_env = lmdb.open(lmdb_dir, map_size=30 * 1024 * 1024 * 1024)

                # Save trajectory data to LMDB
                lmdb_key = f"{scene_name}/{start_cam_idx_i}"
                trajectory_data = {
                    'coverage': coverage_evolution,
                    'X_cam_history': X_cam_history.cpu().numpy(),
                    'V_cam_history': V_cam_history.cpu().numpy(),
                    'points': full_pc.cpu().numpy(),
                    'points_color': full_pc_colors.cpu().numpy()
                }
                save_to_lmdb(lmdb_env, lmdb_key, trajectory_data)

                # Close LMDB
                lmdb_env.close()
                print(f"Closed LMDB database for {scene_name}/{start_cam_idx_i}\n")

                # Cleanup: Keep only imgs folder, delete frames/depths/occupancy folders
                # cleanup_trajectory_folders(training_frames_path, keep_folders=['imgs'])
                # print(f"Finished processing trajectory {start_cam_idx_i}\n")

    print("All trajectories computed.")
