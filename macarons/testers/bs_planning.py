import os
import sys
import ast

import gc
import shutil
from ..utility.macarons_utils import *
from ..utility.tsp_utils import *
from ..utility.utils import count_parameters
import json
import time
# from ..utility.diffusion_utils import *
from ..utility.mcts_planning import *

def evaluate_pose_coverage_gain_mesh_based(pose_idx, camera, mesh, covered_scene, surface_epsilon):
    """
    基于GT mesh rendering计算coverage gain
    
    Args:
        pose_idx: pose索引
        camera: 相机对象  
        mesh: GT mesh
        covered_scene: 已重建点云的scene对象
        surface_epsilon: 表面覆盖阈值距离
    
    Returns:
        unknown_count: 未知点的数量
    """
    # 1. 获取相机参数
    pose, _ = camera.get_pose_from_idx(pose_idx)
    X_cam, V_cam, fov_camera = camera.get_camera_parameters_from_pose(pose)
    
    # 2. 渲染GT mesh得到depth map
    with torch.no_grad():
        gt_images, fragments = camera.renderer(mesh, cameras=fov_camera)
        gt_depth = fragments.zbuf
        gt_mask = gt_depth > 0
    
    # 3. 反投影得到可见的GT surface points
    visible_gt_points = camera.compute_partial_point_cloud(
        depth=gt_depth, 
        mask=gt_mask,
        fov_cameras=fov_camera,
        gathering_factor=1.0
    )
    
    if len(visible_gt_points) == 0:
        return 0
    
    # 4. 获取covered点云
    covered_points = covered_scene.return_entire_pt_cloud(return_features=False)
    
    if len(covered_points) == 0:
        # 没有covered点，所有可见GT点都是未知的
        return len(visible_gt_points)
    
    # 5. 计算未被覆盖的GT点数量
    # 使用cell-based方法优化计算
    unknown_count = 0
    visible_gt_in_box, _ = covered_scene.get_pts_in_bounding_box(visible_gt_points, return_mask=True)
    
    if len(visible_gt_in_box) == 0:
        return 0
        
    involved_cells = covered_scene.get_englobing_cells(visible_gt_in_box, list=True)
    
    for cell_idx in involved_cells:
        key = str(cell_idx)
        
        # 获取该cell中的covered points
        covered_cell = covered_scene.cells[key]
        covered_pts_in_cell = covered_cell.cell_pts
        
        # 获取该cell中的可见GT points
        cell_gt_points, _ = covered_scene.get_pts_in_bounding_box(visible_gt_points, return_mask=True)
        if len(cell_gt_points) == 0:
            continue
            
        # 筛选出真正在这个cell内的GT points
        cell_indices_for_gt = covered_scene.get_cells_for_each_pt(cell_gt_points)
        current_cell_tensor = torch.tensor(cell_idx, device=covered_scene.device)
        mask_in_current_cell = (cell_indices_for_gt == current_cell_tensor).all(dim=1)
        gt_in_this_cell = cell_gt_points[mask_in_current_cell]
        
        if len(gt_in_this_cell) == 0:
            continue
        
        if len(covered_pts_in_cell) > 0:
            # 计算距离
            distances = torch.cdist(gt_in_this_cell.float(), covered_pts_in_cell.float()).min(dim=1)[0]
            uncovered_mask = distances > surface_epsilon
            unknown_count += uncovered_mask.sum().item()
        else:
            # 该cell没有covered点，所有GT点都是未知的
            unknown_count += len(gt_in_this_cell)
    
    return unknown_count


def find_best_pose_by_mesh_rendering(candidate_poses, camera, mesh, covered_scene, surface_epsilon):
    """
    找到未知点数量最多的pose
    """
    all_scores = []
    
    for pose_idx in candidate_poses:
        unknown_count = evaluate_pose_coverage_gain_mesh_based(
            pose_idx, camera, mesh, covered_scene, surface_epsilon
        )
        all_scores.append(unknown_count)
    
    best_pose_idx = np.argmax(all_scores) if all_scores else 0
    return best_pose_idx, all_scores


def load_current_frame_perfect_depth(camera, device):
    """
    专门为perfect depth优化的单帧加载函数
    直接替代复杂的load_images_for_depth_model流程
    """
    current_frame_nb = camera.n_frames_captured - 1
    frame_path = os.path.join(camera.save_dir_path, str(current_frame_nb) + '.pt')
    
    if not os.path.exists(frame_path):
        raise FileNotFoundError(f"当前帧文件不存在: {frame_path}")
    
    frame_dict = torch.load(frame_path, map_location=device)
    
    # 直接返回当前帧的所有需要信息
    return {
        'rgb': frame_dict['rgb'],           # (1, H, W, 3)
        'zbuf': frame_dict['zbuf'],         # (1, H, W, 1) - GT深度!
        'mask': frame_dict['mask'],         # (1, H, W, 1)
        'R': frame_dict['R'],               # (1, 3, 3)
        'T': frame_dict['T'],               # (1, 3)
        'zfar': camera.zfar
    }

def apply_perfect_depth_simple(frame_data, device, use_error_mask=True):
    """
    简化版深度处理，专门用于perfect depth
    直接替代复杂的apply_depth_model函数
    """
    # 直接使用GT深度，无需任何网络推理
    images = frame_data['rgb']
    zbuf = frame_data['zbuf'] 
    mask = frame_data['mask'].bool()
    R = frame_data['R']
    T = frame_data['T']
    
    # 深度就是GT zbuf
    depth = torch.clamp(zbuf, min=0.5, max=750.0)  # 使用zfar限制
    
    # 简单的error mask (如果需要的话)
    if use_error_mask:
        # 对于perfect depth，error mask可以就是原始mask
        error_mask = mask
    else:
        error_mask = torch.ones_like(mask)
    
    return depth, mask, error_mask, R, T


def clear_folder(folder_path):
    """
    清空指定文件夹中的所有内容
    
    参数:
    folder_path (str): 需要清空的文件夹路径
    """
    if os.path.exists(folder_path):
        # 遍历文件夹中的所有文件和子文件夹
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                # 如果是文件，则删除
                os.remove(item_path)
            elif os.path.isdir(item_path):
                # 如果是文件夹，则递归删除
                shutil.rmtree(item_path)
        print(f"文件夹 {folder_path} 已清空")
    else:
        print(f"文件夹 {folder_path} 不存在")

dir_path = os.path.abspath(os.path.dirname(__file__))
# data_path = os.path.join(dir_path, "../../../../../../datasets/rgb")
data_path = os.path.join(dir_path, "../../data/scenes")
results_dir = os.path.join(dir_path, "../../results/scene_exploration")
weights_dir = os.path.join(dir_path, "../../weights/macarons")
configs_dir = os.path.join(dir_path, "../../configs/macarons")


def create_points_to_look_at(X_cam, V_cam, camera_size):
    rays = - get_cartesian_coords(r=torch.ones(len(V_cam), 1, device=V_cam.device),
                                  elev=-1 * V_cam[:, 0].view(-1, 1),
                                  azim=180. + V_cam[:, 1].view(-1, 1),
                                  in_degrees=True)

    cam_pts = X_cam.view(-1, 3) + camera_size * rays
    cam_pts = cam_pts.view(-1, 3)

    return cam_pts


def convert_vector_to_blender(vec):
    new_vec = 0 + vec.cpu().numpy()
    alt_vec = np.copy(new_vec)
    new_vec[..., 1], new_vec[..., 2] = -alt_vec[..., 2], alt_vec[..., 1]
    return new_vec


def convert_blender_to_vector(vec):
    new_vec = np.array(vec)
    alt_vec = np.copy(new_vec)
    new_vec[..., 1], new_vec[..., 2] = alt_vec[..., 2], -alt_vec[..., 1]
    return new_vec


def create_blender_curves(params, X_cam_history, V_cam_history, cam_size=10, jump_poses=1, mirrored_pose=False):
    camera_X = convert_vector_to_blender(X_cam_history[params.n_interpolation_steps::jump_poses])
    camera_look = create_points_to_look_at(X_cam_history[params.n_interpolation_steps::jump_poses],
                                           V_cam_history[params.n_interpolation_steps::jump_poses],
                                           camera_size=cam_size * params.scene_scale_factor)
    camera_look = convert_vector_to_blender(camera_look)

    if mirrored_pose:
        camera_X[..., params.axis_to_mirror] = -1. * camera_X[..., params.axis_to_mirror]
        camera_look[..., params.axis_to_mirror] = -1. * camera_look[..., params.axis_to_mirror]

    camera_X = camera_X / params.scene_scale_factor
    camera_look = camera_look / params.scene_scale_factor

    return camera_X.tolist(), camera_look.tolist()


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
                      mesh, start_cam_idx,
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
    neighbor_indices = camera.get_neighboring_poses(pose_idx=start_cam_idx)
    valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)
    # first_cam_idx = valid_neighbors[np.random.randint(low=0, high=len(valid_neighbors))]
    first_cam_idx = valid_neighbors[-1]

    # Select a random, valid camera pose as starting pose
    camera.initialize_camera(start_cam_idx=first_cam_idx)

    # Capture initial image
    camera.capture_image(mesh)

    # We capture images along the way
    interpolation_step = 1
    for i in range(camera.n_interpolation_steps):
        camera.update_camera(start_cam_idx, interpolation_step=interpolation_step)
        camera.capture_image(mesh)
        interpolation_step += 1

    return camera


def compute_bs_trajectory(
    params, macarons, camera, gt_scene, surface_scene, 
    proxy_scene, covered_scene, mesh, mesh_for_check, device, settings,
    occupancy_threshold=0.5, test_resolution=0.05,
    use_perfect_depth_map=False, compute_collision=False):
    """
    完整的基于双层渲染的未知像素最大化轨迹计算
    
    完全复现原始compute_trajectory的所有处理步骤，并集成双层渲染视点选择
    """
    splited_pose_space_idx = camera.generate_new_splited_dict()
    splited_pose_space = generate_key_value_splited_dict(splited_pose_space_idx)

    macarons.eval()

    collision_free_list = []

    for key, values in splited_pose_space.items():
        ray_origin = values.cpu().numpy()
        # intersections = mesh_for_check.ray.intersects_any(ray_origins=[ray_origin], ray_directions=[np.array([0.0, 1.0, 0.0])])
        # # if intersections[0]:
        # #     continue
        collision_free_list.append(ray_origin)

    collision_free_set = set(tuple(v) for v in collision_free_list)
    full_pc = torch.zeros(0, 3, device=device)
    full_pc_colors = torch.zeros(0, 3, device=device)
    full_pc_idx = torch.zeros(0, 1, device=device)
    coverage_evolution = []
    t0 = time.time()

    traj_index = None
    full_trajectory = []

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
            'X_cam': X_cam,  #正确传递相机位置
            'current_frame': current_frame
        }
            
    pose_i = 0
    gt_pc, gt_colors = gt_scene.return_entire_pt_cloud(return_features= True)
    n_sample = int(gt_pc.shape[0]*0.5)
    indices = torch.randperm(gt_pc.shape[0])[:n_sample]
    downsampled = gt_pc[indices]

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

        #  一次性处理当前帧
        frame_data = process_current_frame()
        if frame_data is None:
            continue

        # 更新场景（一次性完成）
        part_pc_features = torch.zeros(len(frame_data['part_pc']), 1, device=device)
        covered_scene.fill_cells(frame_data['part_pc'], features=part_pc_features)
        surface_scene.fill_cells(frame_data['part_pc'], features=part_pc_features)
        full_pc = torch.vstack((full_pc, frame_data['part_pc']))
        full_pc_colors = torch.vstack((full_pc_colors, frame_data['part_pc_features']))
        part_pc_idx = torch.full((frame_data['part_pc'].shape[0], 1), pose_i, device=device)
        full_pc_idx = torch.vstack((full_pc_idx, part_pc_idx))

        # 正确更新代理点，传入正确的X_cam
        if frame_data['fov_proxy_mask'].any():
            fov_proxy_indices = proxy_scene.get_proxy_indices_from_mask(frame_data['fov_proxy_mask'])
            proxy_scene.fill_cells(frame_data['fov_proxy_points'], 
                                 features=fov_proxy_indices.view(-1, 1))
            
            # 传入正确的相机位置
            proxy_scene.update_proxy_view_states(
                camera, frame_data['fov_proxy_mask'],
                signed_distances=frame_data['sgn_dists'],
                distance_to_surface=None, 
                X_cam=frame_data['X_cam']  # 使用正确的相机位置，而不是None
            )
            
            proxy_scene.update_proxy_supervision_occ(
                frame_data['fov_proxy_mask'], frame_data['sgn_dists'], 
                tol=params.carving_tolerance
            )
            proxy_scene.update_proxy_out_of_field(frame_data['fov_proxy_mask'])

        surface_scene.set_all_features_to_value(value=1.)

        # 计算覆盖率
        current_coverage = gt_scene.scene_coverage(
            covered_scene, surface_epsilon=2 * test_resolution * params.scene_scale_factor
        )
        if pose_i % 10 == 0:
            print("current coverage:", current_coverage)
        coverage_evolution.append(current_coverage[0].item() if current_coverage[0] != 0. else 0.)

        # 可视化splited_pose_space的locations（每个value）作为蓝色立方体和GT点云
        # if pose_i % 10 == 0:  # 每10步可视化一次
        # all_cubes = []
        # for location in splited_pose_space.values():
        #     cube_size = 1.0
        #     cube_density = 3
        #     offsets = torch.linspace(-cube_size/2, cube_size/2, cube_density, device=device)
        #     for dx in offsets:
        #         for dy in offsets:
        #             for dz in offsets:
        #                 cube_point = location + torch.tensor([dx, dy, dz], device=device)
        #                 all_cubes.append(cube_point)

        # if all_cubes:
        #     # 组合可视化：GT点云 + splited_pose_space locations的蓝色立方体
        #     all_vis_points = torch.cat([downsampled, torch.stack(all_cubes)], dim=0)
        #     cube_colors = torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(len(all_cubes), 1)  # 蓝色
        #     all_vis_colors = torch.cat([downsampled_colors, cube_colors], dim=0)

        #     vis = plot_point_cloud(all_vis_points, all_vis_colors,
        #                             name=f"BS Planning Step {pose_i}: GT Points + Blue Pose Space Locations")
        #     vis.show()
        # torch.tensor()

        # if pose_i >= params.n_poses_in_trajectory:
        #     break

        # 占用概率场预测
        # with torch.no_grad():
        #     X_world, view_harmonics, occ_probs = compute_scene_occupancy_probability_field(
        #         params, macarons.scone, camera, surface_scene, proxy_scene, device
        #     )

        # # print(11111111)
        # # print(X_world.shape)

        # # ========== 双层渲染视点选择（核心创新，与原始代码的唯一区别）==========
        
        # # 分离已知和未知点
        # covered_points = covered_scene.return_entire_pt_cloud(return_features=False)
        # occupancy_mask = occ_probs.squeeze() > occupancy_threshold
        # occupancy_points = X_world[occupancy_mask]

        # fig = plot_point_cloud(occupancy_points, torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(occupancy_points.shape[0], 1))
        # fig.show()
        # ========== TSP Guided Beam Search 实现 ==========
        # if pose_i == 0:
        #     # 定义聚类中心和TSP路径
        #     cluster_centers = torch.tensor([
        #         [-25.8, 26.5, 23.0],   # 聚类0
        #         [20.9, 25.5, 23.7],    # 聚类1
        #         [61.6, 24.2, -27.5],   # 聚类2
        #         [-15.2, 27.5, -21.3],  # 聚类3
        #         [-70.6, 22.2, -15.8],  # 聚类4
        #         [59.1, 24.3, 20.2],    # 聚类5
        #         [20.4, 22.9, -14.5],   # 聚类6
        #         [-75.0, 34.4, 17.7]    # 聚类7
        #     ], device=device)

        #     # 动态计算TSP路径
        #     all_locations = torch.cat([camera.X_cam_history[0].view(1, 3), cluster_centers], dim=0)
        #     n_points = len(all_locations)

        #     # 计算距离矩阵
        #     distance_matrix = torch.cdist(all_locations, all_locations)

        #     # 2-opt TSP求解
        #     def solve_tsp_2opt(dist_matrix):
        #         n = dist_matrix.shape[0]
        #         # 贪心初始解
        #         visited = [False] * n
        #         path = [0]
        #         visited[0] = True
        #         current = 0

        #         for _ in range(n - 1):
        #             next_node = -1
        #             min_dist = float('inf')
        #             for j in range(n):
        #                 if not visited[j] and dist_matrix[current, j] < min_dist:
        #                     min_dist = dist_matrix[current, j]
        #                     next_node = j
        #             if next_node != -1:
        #                 visited[next_node] = True
        #                 path.append(next_node)
        #                 current = next_node

        #         # 2-opt改进
        #         def calculate_distance(p):
        #             return sum(dist_matrix[p[i], p[i+1]] for i in range(len(p)-1))

        #         best_path = path
        #         best_dist = calculate_distance(best_path)

        #         for iteration in range(50):  # 限制迭代次数
        #             improved = False
        #             for i in range(1, len(best_path) - 1):
        #                 for j in range(i + 1, len(best_path)):
        #                     new_path = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]
        #                     new_dist = calculate_distance(new_path)
        #                     if new_dist < best_dist:
        #                         best_path = new_path
        #                         best_dist = new_dist
        #                         improved = True
        #                         break
        #                 if improved:
        #                     break
        #             if not improved:
        #                 break

        #         return [x-1 for x in best_path[1:]]  # 去掉起始点，并调整索引到cluster_centers

        #     tsp_path = solve_tsp_2opt(distance_matrix.cpu())
        #     current_target_idx = 0  # 当前目标聚类索引
        #     target_threshold = 10.0  # 到达目标的距离阈值

        #     print(f"TSP路径: {tsp_path}")
        #     print(f"聚类中心: {cluster_centers}")

        #     # 初始化GT点和covered_mask
        #     gt_points = downsampled
        #     covered_mask = torch.zeros(len(gt_points), dtype=torch.bool, device=device)

        #     # 添加当前相机位置作为初始覆盖
        #     current_X_cam = camera.X_cam_history[0]
        #     current_V_cam = camera.V_cam_history[0]
        #     X_cam = current_X_cam.view(1, 3)
        #     V_cam = current_V_cam.view(1, 2)
        #     R_cam, T_cam = get_camera_RT(X_cam, V_cam)
        #     current_fov_camera = FoVPerspectiveCameras(R=R_cam, T=T_cam, zfar=camera.zfar, device=device)

        #     # 渲染并更新初始covered_mask
        #     with torch.no_grad():
        #         _, fragments = camera.renderer(mesh, cameras=current_fov_camera)
        #         current_depth_map = fragments.zbuf[..., 0][0]
        #         current_visible_mask = camera.check_point_visibility_from_depth(gt_points, current_fov_camera, current_depth_map)
        #         covered_mask = covered_mask | current_visible_mask

        #     print(f"初始覆盖点数: {covered_mask.sum().item()}/{len(gt_points)}")

        #     # 初始化TSP guided beam search
        #     initial_pose_idx = camera.cam_idx
        #     beams = [{
        #         'trajectory': [],
        #         'covered_mask': covered_mask.clone(),
        #         'score': covered_mask.sum().item(),
        #         'current_pose_idx': initial_pose_idx
        #     }]
        #     beam_width = 5
        #     guidance_weight = 0.5  # TSP引导权重

        #     def compute_direction_reward(camera_pos, target_pos):
        #         """计算朝向目标的方向奖励"""
        #         distance = torch.norm(camera_pos - target_pos)
        #         # 距离越近奖励越高，使用指数衰减
        #         return torch.exp(-distance / 50.0).item() * 1000

        #     def get_camera_position_from_pose_idx(pose_idx):
        #         """从pose索引获取相机位置"""
        #         pose, _ = camera.get_pose_from_idx(pose_idx)
        #         X_cam, _, _ = camera.get_camera_parameters_from_pose(pose)
        #         return X_cam[0]  # 返回3D位置

        #     # TSP guided beam search主循环
        #     for bs_i in range(params.n_poses_in_trajectory+1):
        #         print(f"TSP Guided Beam Search step {bs_i + 1}/{params.n_poses_in_trajectory}")

        #         # 检查是否需要切换目标
        #         if current_target_idx < len(tsp_path):
        #             current_target_cluster_idx = tsp_path[current_target_idx]
        #             current_target_pos = cluster_centers[current_target_cluster_idx]

        #             # 检查最佳beam是否接近当前目标
        #             best_beam_pos = get_camera_position_from_pose_idx(beams[0]['current_pose_idx'])
        #             distance_to_target = torch.norm(best_beam_pos - current_target_pos).item()

        #             if distance_to_target < target_threshold and current_target_idx < len(tsp_path) - 1:
        #                 current_target_idx += 1
        #                 if current_target_idx < len(tsp_path):  # 确保不越界
        #                     print(f"切换到下一个目标: 聚类{tsp_path[current_target_idx]} at {cluster_centers[tsp_path[current_target_idx]]}")
        #                 else:
        #                     print("已完成一轮TSP循环")

        #             print(f"当前目标: 聚类{current_target_cluster_idx} at {current_target_pos}, 距离: {distance_to_target:.2f}")

        #         all_candidates = []

        #         # 对每个beam进行扩展
        #         for beam in beams:
        #             neighbor_indices = camera.get_neighboring_poses(pose_idx=beam['current_pose_idx'])
        #             valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)

        #             rendering_candidate = []
        #             idx_candidate = []

        #             # 收集有效的邻居poses
        #             for row in valid_neighbors:
        #                 neighbor_pose, _ = camera.get_pose_from_idx(row)
        #                 X_neighbor, V_neighbor, fov_neighbor = camera.get_camera_parameters_from_pose(neighbor_pose)

        #                 # 碰撞检测
        #                 ray_origin = X_neighbor[0].cpu().numpy()
        #                 ray_direction = np.array([0.0, 1.0, 0.0])
        #                 intersections = mesh_for_check.ray.intersects_any(
        #                     ray_origins=[ray_origin],
        #                     ray_directions=[ray_direction]
        #                 )
        #                 if intersections[0]:
        #                     continue

        #                 rendering_candidate.append(fov_neighbor)
        #                 idx_candidate.append(row)

        #             if len(rendering_candidate) == 0:
        #                 continue

        #             # 批量渲染
        #             batch_size = 20
        #             all_depths = []

        #             for batch_start in range(0, len(rendering_candidate), batch_size):
        #                 batch_end = min(batch_start + batch_size, len(rendering_candidate))
        #                 batch_cameras = rendering_candidate[batch_start:batch_end]

        #                 if len(batch_cameras) == 0:
        #                     continue

        #                 from pytorch3d.renderer import join_cameras_as_batch
        #                 batched_cam = join_cameras_as_batch(batch_cameras)
        #                 batch_meshes = mesh.extend(len(batch_cameras))
        #                 _, fragments = camera.renderer(batch_meshes, cameras=batched_cam)
        #                 batch_depths = fragments.zbuf[..., 0]
        #                 all_depths.append(batch_depths)

        #             if len(all_depths) > 0:
        #                 batch_depths = torch.cat(all_depths, dim=0)
        #             else:
        #                 continue

        #             # 计算每个候选pose的TSP guided score
        #             for j, pose_idx in enumerate(idx_candidate):
        #                 depth_map = batch_depths[j]

        #                 # 计算coverage gain
        #                 visible_mask = camera.check_point_visibility_from_depth(
        #                     gt_points, rendering_candidate[j], depth_map
        #                 )
        #                 new_coverage_mask = visible_mask & (~beam['covered_mask'])
        #                 coverage_gain = new_coverage_mask.sum().item()

        #                 # 计算TSP方向奖励
        #                 camera_pos = get_camera_position_from_pose_idx(pose_idx)
        #                 if current_target_idx < len(tsp_path):
        #                     target_cluster_idx = tsp_path[current_target_idx]
        #                     target_pos = cluster_centers[target_cluster_idx]
        #                     direction_reward = compute_direction_reward(camera_pos, target_pos)
        #                 else:
        #                     direction_reward = 0

        #                 # 组合分数：coverage gain + TSP引导
        #                 total_score = coverage_gain + guidance_weight * direction_reward

        #                 # 更新coverage mask
        #                 new_covered_mask = beam['covered_mask'] | visible_mask
        #                 new_trajectory = beam['trajectory'] + [pose_idx]

        #                 all_candidates.append({
        #                     'trajectory': new_trajectory,
        #                     'covered_mask': new_covered_mask,
        #                     'score': total_score,
        #                     'coverage_gain': coverage_gain,
        #                     'direction_reward': direction_reward,
        #                     'current_pose_idx': pose_idx
        #                 })

        #         # 选择top-k候选
        #         if len(all_candidates) == 0:
        #             print("No valid candidates found!")
        #             break

        #         all_candidates.sort(key=lambda x: x['score'], reverse=True)
        #         beams = all_candidates[:beam_width]

        #         print(f"Step {bs_i + 1}: Best score = {beams[0]['score']:.1f} (coverage: {beams[0]['coverage_gain']}, direction: {beams[0]['direction_reward']:.1f})")

        #     # 选择最佳轨迹
        #     if len(beams) > 0:
        #         best_beam = beams[0]
        #         best_trajectory = best_beam['trajectory']

        #         print(f"TSP Guided Best trajectory: {best_trajectory}")
        #         print(f"Total steps planned: {len(best_trajectory)}")

        #         traj_index = 0
        #     else:
        #         print("No valid trajectory found!")
        #         break
#######################################################################################################################
        # if pose_i == 0:
        #     # Greedy轨迹计算
        #     gt_points = downsampled
        #     covered_mask = torch.zeros(len(gt_points), dtype=torch.bool, device=device)

        #     # 初始化覆盖
        #     current_X_cam = camera.X_cam_history[0]
        #     current_V_cam = camera.V_cam_history[0]
        #     X_cam = current_X_cam.view(1, 3)
        #     V_cam = current_V_cam.view(1, 2)
        #     R_cam, T_cam = get_camera_RT(X_cam, V_cam)
        #     current_fov_camera = FoVPerspectiveCameras(R=R_cam, T=T_cam, zfar=camera.zfar, device=device)

        #     with torch.no_grad():
        #         _, fragments = camera.renderer(mesh, cameras=current_fov_camera)
        #         current_depth_map = fragments.zbuf[..., 0][0]
        #         current_visible_mask = camera.check_point_visibility_from_depth(gt_points, current_fov_camera, current_depth_map)
        #         covered_mask = covered_mask | current_visible_mask

        #     print(f"初始覆盖: {covered_mask.sum().item()}/{len(gt_points)}")

        #     # 计算完整greedy轨迹
        #     best_trajectory = []
        #     current_pose_idx = camera.cam_idx

        #     for step in range(params.n_poses_in_trajectory+1):
        #         neighbor_indices = camera.get_neighboring_poses(pose_idx=current_pose_idx)
        #         valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)

        #         if len(valid_neighbors) == 0:
        #             break

        #         best_gain, best_idx = -1, None
        #         for neighbor_idx in valid_neighbors:
        #             neighbor_pose, _ = camera.get_pose_from_idx(neighbor_idx)
        #             X_neighbor, V_neighbor, fov_neighbor = camera.get_camera_parameters_from_pose(neighbor_pose)

        #             # 碰撞检测
        #             ray_origin = X_neighbor[0].cpu().numpy()
        #             intersections = mesh_for_check.ray.intersects_any(
        #                 ray_origins=[ray_origin], ray_directions=[np.array([0.0, 1.0, 0.0])]
        #             )
        #             if intersections[0]:
        #                 continue

        #             # 计算coverage gain
        #             with torch.no_grad():
        #                 _, fragments = camera.renderer(mesh, cameras=fov_neighbor)
        #                 depth_map = fragments.zbuf[..., 0][0]
        #                 visible_mask = camera.check_point_visibility_from_depth(gt_points, fov_neighbor, depth_map)
        #                 gain = (visible_mask & (~covered_mask)).sum().item()

        #             if gain > best_gain:
        #                 best_gain = gain
        #                 best_idx = neighbor_idx
        #                 best_visible_mask = visible_mask

        #         if best_idx is not None:
        #             best_trajectory.append(best_idx)
        #             covered_mask = covered_mask | best_visible_mask
        #             current_pose_idx = best_idx
        #             print(f"Greedy step {step+1}: pose {best_idx}, gain={best_gain}")
        #         else:
        #             break

        #     print(f"Greedy轨迹完成: {len(best_trajectory)} poses, 最终覆盖: {covered_mask.sum().item()}/{len(gt_points)}")
        #     traj_index = 0

        # 计算所有无碰撞poses的全局可见点数
        if pose_i == 0:
            from pytorch3d.renderer import join_cameras_as_batch
            global_visible_mask = torch.zeros(len(downsampled), dtype=torch.bool, device=device)
            all_poses = []

            for pose_vec in camera.pose_space.values():
                if tuple(pose_vec[:3].cpu().numpy()) in collision_free_set:
                    X, V = pose_vec[:3].view(1, 3), pose_vec[3:].view(1, 2)
                    R, T = get_camera_RT(X, V)
                    all_poses.append(FoVPerspectiveCameras(R=R, T=T, zfar=camera.zfar, device=device))

            # 批量渲染
            for i in range(0, len(all_poses), 5):
                batch = all_poses[i:i+5]
                batched_cam = join_cameras_as_batch(batch)
                _, frags = camera.renderer(mesh.extend(len(batch)), cameras=batched_cam)
                for j, cam in enumerate(batch):
                    global_visible_mask |= camera.check_point_visibility_from_depth(
                        downsampled, cam, frags.zbuf[j, ..., 0])
            ratio = global_visible_mask.sum().item()/len(downsampled)
            print(f"全局可见点数: ", ratio)
        break

        # t1 = time.time()
        # if pose_i == 0:
        #     gt_points = downsampled  # 使用已经定义的downsampled
        #     covered_mask = torch.zeros(len(gt_points), dtype=torch.bool, device=device)

        #     # 添加当前相机位置作为初始覆盖
        #     current_X_cam = camera.X_cam_history[0]
        #     current_V_cam = camera.V_cam_history[0]
        #     X_cam = current_X_cam.view(1, 3)
        #     V_cam = current_V_cam.view(1, 2)
        #     R_cam, T_cam = get_camera_RT(X_cam, V_cam)
        #     current_fov_camera = FoVPerspectiveCameras(R=R_cam, T=T_cam, zfar=camera.zfar, device=device)

        #     # 渲染并更新初始covered_mask
        #     with torch.no_grad():
        #         _, fragments = camera.renderer(mesh, cameras=current_fov_camera)
        #         current_depth_map = fragments.zbuf[..., 0][0]
        #         current_visible_mask = camera.check_point_visibility_from_depth(gt_points, current_fov_camera, current_depth_map)
        #         covered_mask = covered_mask | current_visible_mask

        #     print(f"初始覆盖点数: {covered_mask.sum().item()}/{len(gt_points)}")

        #     # 初始化beam search
        #     initial_pose_idx = camera.cam_idx  # 获取当前相机pose索引
        #     beams = [{'trajectory': [], 'covered_mask': covered_mask.clone(), 'score': covered_mask.sum().item(), 'current_pose_idx': initial_pose_idx}]
        #     beam_width = 20

        #     for bs_i in range(params.n_poses_in_trajectory+1):
        #         print(f"Beam search step {bs_i + 1}/{params.n_poses_in_trajectory}")

        #         all_candidates = []

        #         # 对每个beam进行扩展
        #         for beam in beams:
        #             # 获取邻居poses，传入当前beam的pose_idx
        #             neighbor_indices = camera.get_neighboring_poses(pose_idx=beam['current_pose_idx'])
        #             valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)

        #             rendering_candidate = []
        #             idx_candidate = []

        #             # 收集有效的邻居poses
        #             for row in valid_neighbors:
        #                 neighbor_pose, _ = camera.get_pose_from_idx(row)
        #                 X_neighbor, V_neighbor, fov_neighbor = camera.get_camera_parameters_from_pose(neighbor_pose)

        #                 # 碰撞检测
        #                 ray_origin = X_neighbor[0].cpu().numpy()
        #                 ray_tuple = tuple(ray_origin)
        #                 if ray_tuple not in collision_free_set:
        #                     continue

        #                 rendering_candidate.append(fov_neighbor)
        #                 idx_candidate.append(row)

        #             if len(rendering_candidate) == 0:
        #                 continue

        #             # 批量渲染所有候选poses，batch size为20
        #             batch_size = 20
        #             all_depths = []

        #             for batch_start in range(0, len(rendering_candidate), batch_size):
        #                 t1 = time.time()
        #                 batch_end = min(batch_start + batch_size, len(rendering_candidate))
        #                 batch_cameras = rendering_candidate[batch_start:batch_end]

        #                 if len(batch_cameras) == 0:
        #                     continue

        #                 from pytorch3d.renderer import join_cameras_as_batch
        #                 batched_cam = join_cameras_as_batch(batch_cameras)
        #                 batch_meshes = mesh.extend(len(batch_cameras))
        #                 _, fragments = camera.renderer(batch_meshes, cameras=batched_cam)
        #                 batch_depths = fragments.zbuf[..., 0]
                        
        #                 all_depths.append(batch_depths)
        #                 t2 = time.time()
        #                 print(t2-t1)
        #                 torch.tensor()

        #             # 合并所有batch的结果
        #             if len(all_depths) > 0:
        #                 batch_depths = torch.cat(all_depths, dim=0)
        #             else:
        #                 continue  # 如果没有有效候选，跳过这个beam

        #             # 计算每个候选pose的coverage gain
        #             for j, pose_idx in enumerate(idx_candidate):
        #                 depth_map = batch_depths[j]

        #                 # 计算这个pose能看到的点
        #                 visible_mask = camera.check_point_visibility_from_depth(
        #                     gt_points, rendering_candidate[j], depth_map
        #                 )

        #                 # 计算新增coverage（只考虑beam中还没被覆盖的点）
        #                 new_coverage_mask = visible_mask & (~beam['covered_mask'])
        #                 coverage_gain = new_coverage_mask.sum().item()

        #                 # 创建新的候选
        #                 new_covered_mask = beam['covered_mask'] | visible_mask
        #                 new_trajectory = beam['trajectory'] + [pose_idx]
        #                 new_score = new_covered_mask.sum().item()

        #                 all_candidates.append({
        #                     'trajectory': new_trajectory,
        #                     'covered_mask': new_covered_mask,
        #                     'score': new_score,
        #                     'coverage_gain': coverage_gain,
        #                     'current_pose_idx': pose_idx
        #                 })

        #         # 选择top-k候选作为新的beams
        #         if len(all_candidates) == 0:
        #             print("No valid candidates found!")
        #             break

        #         all_candidates.sort(key=lambda x: x['score'], reverse=True)
        #         beams = all_candidates[:beam_width]

        #         print(f"Step {bs_i + 1}: Best score = {beams[0]['score']/len(gt_points)}, Coverage gain = {beams[0].get('coverage_gain', 0)}")
        #         print(f"Top {min(beam_width, len(all_candidates))} beams selected from {len(all_candidates)} candidates")

        #     # 选择最佳轨迹
        #     if len(beams) > 0:
        #         best_beam = beams[0]
        #         best_trajectory = best_beam['trajectory']
        #         final_coverage = best_beam['score']

        #         print(f"Best trajectory found: {best_trajectory}")
        #         print(f"Final coverage: {final_coverage}/{len(gt_points)} points")

        #         # 设置轨迹供后续使用
        #         # full_trajectory = [str(pose_idx) for pose_idx in best_trajectory]
        #         traj_index = 0
        #         # print(full_trajectory)
        #     else:
        #         print("No valid trajectory found!")
        #         break
        # t2 = time.time()
        # print(t2-t1)
        # # 获取候选位姿
        # # if traj_index is None or traj_index == len(full_trajectory):
        # #     selected_viewpoints = camera.quality_based_adaptive_viewpoint_selection(
        # #                             mesh=mesh,
        # #                             mesh_for_check=mesh_for_check,
        # #                             gt_point_cloud= gt_scene.return_entire_pt_cloud(return_features=False)
        # #                         )

        # # # 移动到下一个位置（与原始代码完全一致）


        # next_idx = best_trajectory[traj_index]
        # interpolation_step = 1
        # for i in range(camera.n_interpolation_steps):
        #     camera.update_camera(next_idx, interpolation_step=interpolation_step)
        #     camera.capture_image(mesh)
        #     interpolation_step += 1
        # traj_index += 1
        # pose_i += 1
    

    # pc_plot = plot_point_cloud(full_pc, torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(full_pc.shape[0], 1), name='Filtered reconstructed surface points', 
    #                 point_size=2, max_points=150000, width=800, height=600, cmap='rgb')
    # pc_plot.show()

    print("Trajectory computed in", time.time() - t0, "seconds.")
    print("Coverage Evolution:", coverage_evolution)

    return coverage_evolution, camera.X_cam_history, camera.V_cam_history, full_pc, full_pc_colors, full_pc_idx, ratio

def run_bs_test(params_name,
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
    if load_json:
        with open(results_json_path, "r") as read_content:
            dict_to_save = json.load(read_content)
    else:
        dict_to_save = {}

    print("\nModel path:", model_name)
    print("\nScore threshold:", params.score_threshold)

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

            dict_to_save[scene_name] = {}

            scene_path = os.path.join(dataloader.dataset.data_path, scene_name)
            mesh_path = os.path.join(scene_path, obj_name)
            segmented_mesh_path = os.path.join(scene_path, 'segmented.obj')

            mirrored_scene = False
            mirrored_axis = None

            # Load mesh
            mesh = load_scene(mesh_path, params.scene_scale_factor, device,
                              mirror=mirrored_scene, mirrored_axis=mirrored_axis)
            
            import trimesh
            mesh_for_check = trimesh.load(mesh_path)

            if isinstance(mesh_for_check, trimesh.Scene):
                # 使用 dump 方法合并场景
                mesh_for_check = mesh_for_check.dump(concatenate=True)
            mesh_for_check.vertices *= params.scene_scale_factor

            print("Mesh Vertices shape:", mesh.verts_list()[0].shape)
            print("Min Vert:", torch.min(mesh.verts_list()[0], dim=0)[0],
                  "\nMax Vert:", torch.max(mesh.verts_list()[0], dim=0)[0])

            # Use memory info to set frames and poses path
            scene_memory_path = os.path.join(scene_path, params.memory_dir_name)
            trajectory_nb = memory.current_epoch % memory.n_trajectories
            training_frames_path = memory.get_trajectory_frames_path(scene_memory_path, trajectory_nb)
            training_poses_path = memory.get_poses_path(scene_memory_path)

            torch.cuda.empty_cache()

            for start_cam_idx_i in range(1): #len(settings.camera.start_positions)
                start_cam_idx = settings.camera.start_positions[start_cam_idx_i]
                print("Start cam index for " + scene_name + ":", start_cam_idx)

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

                clear_folder(training_frames_path)
                camera = setup_test_camera(params, mesh, start_cam_idx, settings, occupied_pose_data,
                                           device, training_frames_path,
                                           mirrored_scene=mirrored_scene, mirrored_axis=mirrored_axis)
                print(camera.X_cam_history[0], camera.V_cam_history[0])

                coverage_evolution, X_cam_history, V_cam_history, full_pc, full_pc_colors, full_pc_idx, ratio = compute_bs_trajectory(params, macarons,
                                                                                      camera,
                                                                                      gt_scene, surface_scene,
                                                                                      proxy_scene, covered_scene,
                                                                                      mesh, mesh_for_check,
                                                                                      device,
                                                                                      settings)
                

                # plot_scene_and_tragectory_and_constructed_pt(scene_name=scene_name, params=params, gt_scene=gt_scene,
                #                                             proxy_scene=proxy_scene, macarons=macarons,
                #                                             surface_scene=surface_scene, camera=camera,
                #                                             i_th_scene=i_scene, memory=memory,
                #                                             device=device, results_dir=results_dir)
               

                dict_to_save[scene_name][str(start_cam_idx_i)] = {}
                dict_to_save[scene_name][str(start_cam_idx_i)]["ratio"] = ratio
                # dict_to_save[scene_name][str(start_cam_idx_i)]["coverage"] = coverage_evolution
                # dict_to_save[scene_name][str(start_cam_idx_i)]["X_cam_history"] = X_cam_history.cpu().numpy().tolist()
                # dict_to_save[scene_name][str(start_cam_idx_i)]["V_cam_history"] = V_cam_history.cpu().numpy().tolist()

                # Store pc

                with open(results_json_path, 'w') as outfile:
                    json.dump(dict_to_save, outfile)
                print("Saved data about test losses in", results_json_path)

                # dict_to_save = {'points': full_pc.cpu().numpy().tolist(), 'colors': full_pc_colors.cpu().numpy().tolist(), "idx": full_pc_idx.cpu().numpy().tolist()}
                # reconstructed_json_name = os.path.join(results_dir, "my" + scene_name + '.json')
                # with open(reconstructed_json_name, 'w') as outfile:
                #     json.dump(dict_to_save, outfile)

                # torch.tensor()

    print("All trajectories computed.")
