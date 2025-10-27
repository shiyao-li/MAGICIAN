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
    first_cam_idx = valid_neighbors[np.random.randint(low=0, high=len(valid_neighbors))]

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


def compute_tsp_trajectory(
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
    curriculum_distances = get_curriculum_sampling_distances(params, surface_scene, proxy_scene)
    full_pc = torch.zeros(0, 3, device=device)
    coverage_evolution = []
    t0 = time.time()

    traj_index = None
    full_trajectory = []

    def process_current_frame():
        """统一处理当前帧，修复X_cam参数"""
        try:
            current_frame = load_current_frame_perfect_depth(camera, device)
            depth, mask, error_mask, R, T = apply_perfect_depth_simple(current_frame, device)
            
            # 获取相机在世界坐标中的位置
            fov_camera = camera.get_fov_camera_from_RT(R_cam=R, T_cam=T)
            X_cam = fov_camera.get_camera_center()  
            
            # 计算点云
            part_pc = camera.compute_partial_point_cloud(
                depth=depth, mask=(mask * error_mask).bool(),
                fov_cameras=fov_camera,
                gathering_factor=params.gathering_factor,
                fov_range=params.sensor_range
            )
            
            # 处理代理点
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
                'fov_proxy_points': fov_proxy_points,
                'fov_proxy_mask': fov_proxy_mask,
                'sgn_dists': sgn_dists,
                'X_cam': X_cam,  #正确传递相机位置
                'current_frame': current_frame
            }
            
        except FileNotFoundError as e:
            print(f"无法处理帧: {e}")
            return None
    pose_i = 0
    # for pose_i in range(params.n_poses_in_trajectory + 1):
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

        start_list = []

        gt_points = gt_scene.return_entire_pt_cloud(return_features=False)
        gt_colors = torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(gt_points.shape[0], 1)

        cube_size = 0.5 # 立方体大小，可调整
        density = 10

        cube_points_list = []
        # print(f"DEBUG: camera x_min = {settings.camera.x_min}")
        # print(f"DEBUG: camera x_max = {settings.camera.x_max}")
        # print(f"DEBUG: pose_l={settings.camera.pose_l}, pose_w={settings.camera.pose_w}, pose_h={settings.camera.pose_h}")
        # print(f"Total items in splited_pose_space: {len(splited_pose_space)}")
        for locc, point in splited_pose_space.items():
            # print(f"Key: {locc}, Point: {point}") 
            # print(point)
            import numpy as np
            # 提取坐标
            x, y, z = point
            # ray_origin = point.cpu().numpy()
            # ray_direction = np.array([0.0, 1.0, 0.0])
            # intersections = mesh_for_check.ray.intersects_any(
            #     ray_origins=[ray_origin],
            #     ray_directions=[ray_direction]
            # )
            # if not intersections[0]:
            #     continue
            
            # 在每个方向上生成点
            offsets = torch.linspace(-cube_size/2, cube_size/2, density, device=point.device)
            
            # 生成立方体内的所有点
            for dx in offsets:
                for dy in offsets:
                    for dz in offsets:
                        cube_points_list.append([x + dx, y + dy, z + dz])
            start_list.append(locc)

        cube_points = torch.tensor(cube_points_list, device=device)

        # 生成颜色（比如蓝色）
        cube_colors = torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(cube_points.shape[0], 1)
        fig = plot_point_cloud(torch.cat([gt_points, cube_points], dim=0), torch.cat([gt_colors, cube_colors], dim=0), name="100 anchor views")
        fig.show()
        torch.tensor()
        # sampled_camera = camera.sample_valid_poses_in_space(mesh, proxy_scene, num_samples=300)
        # # start_list = [tuple(ast.literal_eval(s)) if isinstance(s, str) else tuple(s) for s in start_list]
        # # random.shuffle(start_list)
        # import numpy as np
        # iii = 0
        # final_stat = []
        # for item in sampled_camera:
        #     key_idx = item[-1]
        #     lst = ast.literal_eval(key_idx) if isinstance(key_idx, str) else key_idx
        #     print(lst)

        #     ray_origin = np.array(lst[:3])
        #     ray_direction = np.array([0.0, 1.0, 0.0])
        #     if line_segment_mesh_intersection(ray_origin, ray_direction, mesh_for_check):
        #         continue
            
        #     else:
        #         final_stat.append(lst)
        #         iii += 1
        #     if iii == 5:
        #         print(final_stat)
        #         torch.tensor()
        # torch.tensor()

        # sample = random.sample(start_list, k=min(5, len(start_list)))

        # for item in sample:
        #     print(item)
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

        # neighbor_indices = camera.get_neighboring_poses()
        
        # 获取候选位姿
        # if traj_index is None or traj_index == len(full_trajectory):
        #     selected_viewpoints = camera.quality_based_adaptive_viewpoint_selection(
        #                             mesh=mesh,
        #                             mesh_for_check=mesh_for_check,
        #                             gt_point_cloud= gt_scene.return_entire_pt_cloud(return_features=False)
        #                         )
        gt_pc, gt_colors = gt_scene.return_entire_pt_cloud(return_features= True)
        # cube_colors = torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(gt_pc.shape[0], 1)
        # vis = plot_point_cloud(gt_pc, cube_colors)
        # vis.show()
        n_sample = int(gt_pc.shape[0] * 0.5)
        indices = torch.randperm(gt_pc.shape[0])[:n_sample]
        downsampled = gt_pc[indices]
        downsampled_colors = gt_colors[indices]
        print(settings.camera.x_min, settings.camera.x_max)
         # 保持原范围[-60°,60°]，15度步长
            #保持原范围[0°,345°]，15度步长

        # current_ppse

        print(camera.X_cam_history[0], camera.V_cam_history[0])

        def sample_5d_vectors(n_samples, device=device):
            """
            在五维空间中随机采样n个向量 - 高效版本
            
            Args:
                n_samples: 采样数量
                device: 设备类型，默认'cuda:0'
            
            Returns:
                tensor: shape为(n_samples, 5)的张量
            """
            # 定义范围 [min, max]
            min_vals = torch.tensor([-80, 0, -30, -60, 0], device=device, dtype=torch.float32)
            max_vals = torch.tensor([70, 40, 30, 60, 360], device=device, dtype=torch.float32)
            
            # 一次性生成所有随机数，然后缩放
            random_samples = torch.rand(n_samples, 5, device=device)
            samples = random_samples * (max_vals - min_vals) + min_vals
            
            return samples

        def greedy_view_selection(camera, mesh, gt_points, mesh_for_check, device, target_coverage=0.95, max_views=50):
            """
            贪心选择相机视角直到达到目标覆盖率
            """
            selected_poses = []
            selected_depth_maps = []
            best_py3d_cameras = []
            covered_mask = torch.zeros(len(gt_points), dtype=torch.bool, device=device)

            # 添加当前相机位置作为第一个anchor pose
            current_X_cam = camera.X_cam_history[0]  # tensor([41.8750, 15.0000, 15.0000], device='cuda:0')
            current_V_cam = camera.V_cam_history[0]  # tensor([  0., 180.], device='cuda:0')
            current_pose = torch.cat([current_X_cam, current_V_cam], dim=0)  # 拼接成5D向量

            print(f"当前相机位置作为第一个anchor: {current_pose}")

            # 计算当前相机的覆盖范围
            X_cam = current_X_cam.view(1, 3)
            V_cam = current_V_cam.view(1, 2)
            R_cam, T_cam = get_camera_RT(X_cam, V_cam)
            current_fov_camera = FoVPerspectiveCameras(R=R_cam, T=T_cam, zfar=camera.zfar, device=device)

            # 渲染当前相机的深度图
            with torch.no_grad():
                _, fragments = camera.renderer(mesh, cameras=current_fov_camera)
                current_depth_map = fragments.zbuf[..., 0][0]  # 第一个相机的深度图

                # 计算当前相机的可见点
                current_visible_mask = camera.check_point_visibility_from_depth(gt_points, current_fov_camera, current_depth_map)
                covered_mask = covered_mask | current_visible_mask

                # 添加到已选择的poses
                selected_poses.append(current_pose)
                selected_depth_maps.append(current_depth_map.clone())
                best_py3d_cameras.append(current_fov_camera)

                current_coverage = covered_mask.sum().item() / len(gt_points)
                print(f"当前相机覆盖: {current_visible_mask.sum().item()} 点")
                print(f"初始覆盖率: {current_coverage*100:.2f}%")

            check_batch_size = 300
            render_batch_size = 20
            sample_size = 50

            # current_coverage和view_count已在上面设置
            view_count = 1  # 已经有了第一个相机

            print(f"目标覆盖率: {target_coverage*100:.1f}%, 最大视图数: {max_views}")
            print(f"GT点总数: {len(gt_points)}")

            while current_coverage < target_coverage and view_count < max_views:
                import numpy as np
                print(f"\n=== 第 {view_count + 1} 轮 ===")
                
                # 采样向量
                vectors = sample_5d_vectors(sample_size, device=device)
                print(f"采样向量数量: {len(vectors)}")
                
                # 第一个循环：射线检测和FOV检查
                fov_passed_data = []
                for i in range(0, len(vectors), check_batch_size):
                    batch_vectors = vectors[i:i+check_batch_size]
                    
                    # 批量射线检测
                    ray_origins = [vec[:3].cpu().numpy() for vec in batch_vectors]
                    ray_directions = np.array([[0.0, 1.0, 0.0]] * len(batch_vectors))
                    
                    intersections = mesh_for_check.ray.intersects_any(
                        ray_origins=ray_origins,
                        ray_directions=ray_directions
                    )
                    
                    # 筛选出没有相交的向量
                    ray_passed_vectors = []
                    for vec, has_intersection in zip(batch_vectors, intersections):
                        if not has_intersection:
                            ray_passed_vectors.append(vec)
                    
                    if not ray_passed_vectors:
                        continue
                    
                    # FOV检查
                    for vec in ray_passed_vectors:
                        X_cam = vec[:3].view(1, 3)
                        V_cam = vec[3:].view(1, 2)
                        R_cam, T_cam = get_camera_RT(X_cam, V_cam)
                        fov_camera = FoVPerspectiveCameras(R=R_cam, T=T_cam, zfar=camera.zfar, device=device)
                        
                        if not camera.is_fov_empty(mesh, fov_camera):
                            fov_passed_data.append((vec, R_cam, T_cam))
                
                print(f"通过FOV检查的向量: {len(fov_passed_data)}")
                
                if not fov_passed_data:
                    print("没有有效的向量，停止采样")
                    break
                
                # 第二个循环：批量渲染深度图并计算增益
                best_pose = None
                best_depth_map = None
                best_gain = 0
                
                for i in range(0, len(fov_passed_data), render_batch_size):
                    render_batch = fov_passed_data[i:i+render_batch_size]
                    
                    batch_R = [data[1] for data in render_batch]
                    batch_T = [data[2] for data in render_batch]
                    render_vectors = [data[0] for data in render_batch]
                    
                    batch_cameras = FoVPerspectiveCameras(
                        R=torch.cat(batch_R, dim=0),
                        T=torch.cat(batch_T, dim=0),
                        zfar=camera.zfar,
                        device=device
                    )
                    # torch.tensor()
                    
                    batch_meshes = mesh.extend(len(render_vectors))
                    with torch.no_grad():
                        _, fragments = camera.renderer(batch_meshes, cameras=batch_cameras)
                        batch_depths = fragments.zbuf[..., 0]
                        
                        for j, vec in enumerate(render_vectors):
                            depth_map = batch_depths[j]
                            
                            # 计算当前视图的可见点
                            visible_mask = camera.check_point_visibility_from_depth(gt_points, batch_cameras[j], depth_map)
                            
                            # 计算新增可见点数量（贪心策略）
                            new_visible_mask = visible_mask & (~covered_mask)
                            new_visible_count = new_visible_mask.sum().item()
                            
                            if new_visible_count > best_gain:
                                best_gain = new_visible_count
                                best_pose = vec
                                best_FoVPerspectiveCameras = batch_cameras[j]
                                best_depth_map = depth_map.clone()
                    
                    # 清理内存
                    del batch_cameras, batch_meshes, fragments, batch_depths
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # 选择最佳pose并更新覆盖率
                if best_pose is not None and best_gain > 0:
                    selected_poses.append(best_pose)
                    selected_depth_maps.append(best_depth_map)
                    best_py3d_cameras.append(best_FoVPerspectiveCameras)
                    
                    # 更新已覆盖的点
                    best_visible_mask = camera.check_point_visibility_from_depth(gt_points, best_FoVPerspectiveCameras, best_depth_map)
                    covered_mask = covered_mask | best_visible_mask
                    
                    current_coverage = covered_mask.sum().item() / len(gt_points)
                    view_count += 1
                    
                    print(f"选择第 {view_count} 个视图:")
                    print(f"  新增可见点: {best_gain}")
                    print(f"  总覆盖点数: {covered_mask.sum().item()}")
                    print(f"  当前覆盖率: {current_coverage*100:.2f}%")
                else:
                    print("没有找到更好的视图，停止搜索")
                    break
            
            print(f"\n=== 完成 ===")
            print(f"选择了 {len(selected_poses)} 个视图")
            print(f"最终覆盖率: {current_coverage*100:.2f}%")
            print(f"覆盖点数: {covered_mask.sum().item()} / {len(gt_points)}")
            
            return selected_poses, selected_depth_maps, covered_mask, best_py3d_cameras

        # # 使用示例
        selected_poses, selected_depth_maps, final_covered_mask, best_py3d_cameras = greedy_view_selection(
            camera=camera,
            mesh=mesh,
            gt_points=downsampled,
            mesh_for_check=mesh_for_check,
            device=device,
            target_coverage=0.95,
            max_views=10
        )
        print(len(selected_poses))
        print(len(best_py3d_cameras))
        print(selected_poses)
        sub_selected_poses = selected_poses[1:6]
        sub_best_py3d_cameras = best_py3d_cameras[1:6]

        novel_X_cam_history = []
        novel_V_cam_history = []

        for sub in range(len(sub_selected_poses)):
            novel_pose = sub_selected_poses[sub].cpu().numpy()
            novel_X_cam_history.append(novel_pose[:3])
            novel_V_cam_history.append(novel_pose[-2:])

            camera.capture_image(mesh, fov_camera=sub_best_py3d_cameras[sub],dir_path='/home/sli/phd_projects/novel_views')


        # 保存为JSON文件
        output_data = {
            "X_cam_history": [x.tolist() for x in novel_X_cam_history],  # 转换为列表
            "V_cam_history": [v.tolist() for v in novel_V_cam_history]   # 转换为列表
        }

        json_path = '/home/sli/phd_projects/novel_views/camera_history.json'
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        print(f"相机历史已保存到: {json_path}")
        print(f"保存了 {len(novel_X_cam_history)} 个视角")


        torch.tensor()

        # # 可视化相机位置、朝向和覆盖结果
        # if selected_poses and len(selected_poses) > 0:
        #     # 1. 相机位置和朝向
        #     camera_positions = []
        #     camera_directions = []
        #     camera_cubes = []

        #     for pose in selected_poses:
        #         # 位置
        #         position = pose[:3]
        #         camera_positions.append(position)

        #         # 修复相机朝向计算：使用与get_camera_RT相同的转换
        #         elev = pose[3]  # elevation角度
        #         azim = pose[4]  # azimuth角度

        #         # 按照get_camera_RT的转换方式计算朝向向量
        #         # rays = - get_cartesian_coords(elev=-1*elev, azim=180+azim)
        #         elev_transformed = -elev
        #         azim_transformed = 180.0 + azim

        #         # 转换为弧度
        #         elev_rad = torch.deg2rad(elev_transformed)
        #         azim_rad = torch.deg2rad(azim_transformed)

        #         # 使用get_cartesian_coords的公式，然后取负值
        #         ray_x = torch.cos(elev_rad) * torch.sin(azim_rad)
        #         ray_y = torch.sin(elev_rad)
        #         ray_z = torch.cos(elev_rad) * torch.cos(azim_rad)

        #         # 取负值（与get_camera_RT中的负号一致）
        #         forward = -torch.stack([ray_x, ray_y, ray_z])

        #         # 朝向终点 (相机看向的方向)
        #         direction_end = position + forward * 15.0
        #         camera_directions.append(direction_end)

        #         # 为每个相机位置创建小立方体
        #         cube_size = 2.0  # 立方体大小
        #         cube_density = 3  # 每个方向的点数

        #         # 生成立方体内的点
        #         cube_points_for_cam = []
        #         offsets = torch.linspace(-cube_size/2, cube_size/2, cube_density, device=device)
        #         for dx in offsets:
        #             for dy in offsets:
        #                 for dz in offsets:
        #                     cube_point = position + torch.tensor([dx, dy, dz], device=device)
        #                     cube_points_for_cam.append(cube_point)

        #         camera_cubes.extend(cube_points_for_cam)

        #     # 转为tensor
        #     cam_pos = torch.stack(camera_positions)
        #     cam_dir = torch.stack(camera_directions)
        #     cam_cubes = torch.stack(camera_cubes) if camera_cubes else torch.empty(0, 3, device=device)

        #     # 2. 覆盖点云分析
        #     covered_points = downsampled[final_covered_mask]
        #     uncovered_points = downsampled[~final_covered_mask]
        #     uncovered_points_colors = downsampled_colors[~final_covered_mask]

        #     print(f"覆盖: {len(covered_points)}, 未覆盖: {len(uncovered_points)}")

        #     # 3. 组合可视化
        #     all_points = torch.cat([
        #         covered_points,      # 绿色 - 已覆盖的GT点
        #         uncovered_points,    # 红色 - 未覆盖的GT点
        #         cam_cubes,          # 蓝色 - 相机位置立方体
        #         cam_dir             # 黄色 - 相机朝向
        #     ], dim=0)

        #     all_colors = torch.cat([
        #         torch.tensor([[0.0, 1.0, 0.0]], device=device).repeat(len(covered_points), 1),    # 绿色
        #         # torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(len(uncovered_points), 1),  # 红色
        #         uncovered_points_colors,
        #         torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(len(cam_cubes), 1),         # 蓝色立方体
        #         torch.tensor([[1.0, 1.0, 0.0]], device=device).repeat(len(cam_dir), 1)            # 黄色
        #     ], dim=0)

        #     vis = plot_point_cloud(all_points, all_colors, name="Coverage Analysis: Green=Covered, Red=Uncovered, Blue=Camera, Yellow=Direction")
        #     vis.show()

        #     # 4. 渲染并保存RGB图像
        #     print("\n=== 渲染相机视图 ===")
        #     for i, pose in enumerate(selected_poses):
        #         X_cam = pose[:3].view(1, 3)
        #         V_cam = pose[3:].view(1, 2)
        #         R_cam, T_cam = get_camera_RT(X_cam, V_cam)

        #         fov_camera = FoVPerspectiveCameras(R=R_cam, T=T_cam, zfar=camera.zfar, device=device)

        #         with torch.no_grad():
        #             rgb_images, fragments = camera.renderer(mesh, cameras=fov_camera)
        #             rgb_image = rgb_images[0]  # 第一张图片

        #             # 转换为numpy并调整为正确的图像格式
        #             rgb_np = rgb_image.cpu().numpy()
        #             rgb_np = (rgb_np * 255).astype('uint8')  # 转为0-255

        #             # 保存图片
        #             from PIL import Image
        #             filename = f"camera_{i+1}_view.png"
        #             Image.fromarray(rgb_np).save(filename)

        #             print(f"保存相机 {i+1} 视图: {filename}")
        #             print(f"  位置: [{pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}]")
        #             print(f"  角度: elev={pose[3]:.1f}°, azim={pose[4]:.1f}°")
        # current_X_cam = camera.X_cam_history[0]  # tensor([41.8750, 15.0000, 15.0000], device='cuda:0')
        # current_V_cam = camera.V_cam_history[0]  # tensor([  0., 180.], device='cuda:0')
        # current_pose = torch.cat([current_X_cam, current_V_cam], dim=0)  # 拼接成5D向量

        # selected_poses = [torch.tensor([ 41.8750,   5.0000,  25.0000,   0.0000, 180.0000], device='cuda:0'), 
        #                   torch.tensor([-74.8786,  33.2691, -27.7092, -40.3994,  35.2206], device='cuda:0'), 
        #                   torch.tensor([ 59.3010,  38.7499,  23.9080, -29.4960, 256.0100], device='cuda:0'), 
        #                   torch.tensor([ 65.6885,  24.5794, -27.2157, -10.0382, 312.8635], device='cuda:0'), 
        #                   torch.tensor([-43.4753,  14.2058,  29.9911,   1.5889,  84.3383], device='cuda:0'), 
        #                   torch.tensor([-28.2510,  38.3194, -24.6252, -44.0513, 102.9686], device='cuda:0'), 
        #                   torch.tensor([-77.4669,  36.4234,  26.1466, -53.4333, 210.2920], device='cuda:0'), 
        #                   torch.tensor([ 53.4078,  39.4984,   1.7619, -40.4095, 225.6408], device='cuda:0'), 
        #                   torch.tensor([ 33.0255,  36.3187, -23.6077, -44.9455,   7.1789], device='cuda:0'), 
        #                   torch.tensor([ 34.3136,  37.1314,  29.5325, -45.8364,  96.6335], device='cuda:0'), 
        #                   torch.tensor([-76.0353,  18.1390, -23.7619, -22.4143, 125.2457], device='cuda:0'), 
        #                   torch.tensor([ 22.4540,  35.8142,  24.7291, -53.5387, 260.7314], device='cuda:0'), 
        #                   torch.tensor([ 62.9424,  36.7619,  28.4662, -42.6243, 177.8295], device='cuda:0'), 
        #                   torch.tensor([ 68.4741,   2.2929,  28.0788,  34.3714, 223.2264], device='cuda:0'), 
        #                   torch.tensor([ 65.0686,  39.0622, -29.3214, -29.9656, 306.3366], device='cuda:0'), 
        #                   torch.tensor([-76.4130,  31.7907,  16.9642,  -5.0738, 145.8578], device='cuda:0'), 
        #                   torch.tensor([ 62.3485,  18.0567,  29.6833, -37.3313, 206.1369], device='cuda:0'), 
        #                   torch.tensor([ 40.7880,   6.6395, -29.1900, -13.7850, 320.7446], device='cuda:0'), 
        #                   torch.tensor([-66.8414,  17.2129,   4.1310, -40.6135, 182.7557], device='cuda:0'), 
        #                   torch.tensor([ 12.2168,  17.2016,  28.8078, -23.7832, 181.7886], device='cuda:0'), 
        #                   torch.tensor([-23.0422,  23.9236, -23.0023, -30.8099, 317.9084], device='cuda:0'), 
        #                   torch.tensor([ -0.8270,  29.9878, -27.7647, -59.2085,  98.5016], device='cuda:0'), 
        #                   torch.tensor([ 58.6603,  37.0512, -20.4541, -27.7854, 310.3539], device='cuda:0'), 
        #                   torch.tensor([-25.3953,  25.4369,  25.6010, -32.9376, 137.7699], device='cuda:0'), 
        #                   torch.tensor([-68.0086,  24.2022,  -4.1648, -11.7049, 106.5766], device='cuda:0'), 
        #                   torch.tensor([ -8.4265,  39.9423,  13.2687, -23.9227, 103.6788], device='cuda:0'), 
        #                   torch.tensor([ 11.2268,   5.2723,  28.9782, -19.4412, 179.0654], device='cuda:0'), 
        #                   torch.tensor([-77.8716,  30.0271, -28.7105,  -4.9306, 123.7825], device='cuda:0'), 
        #                   torch.tensor([ 28.2253,  24.7507,  28.4880, -49.9948, 150.7911], device='cuda:0'), 
        #                   torch.tensor([-59.7686,  10.3671, -14.8666,  17.9513,  10.5413], device='cuda:0'), 
        #                   torch.tensor([ 69.9807,  17.1291, -29.6621, -43.9233, 312.8589], device='cuda:0'), 
        #                   torch.tensor([-15.9444,  36.2579, -16.2055, -39.8862, 150.5810], device='cuda:0'), 
        #                   torch.tensor([ 52.7037,  38.5904,  18.1216, -56.3140, 172.7599], device='cuda:0'), 
        #                   torch.tensor([ 12.5958,  36.3749, -27.7899, -52.4181,  96.9358], device='cuda:0'), 
        #                   torch.tensor([ 20.3029,  15.2869,   8.3182, -40.4492,   4.0692], device='cuda:0'), 
        #                   torch.tensor([ 69.3952,   7.7794,   9.2918, -26.7170, 339.2213], device='cuda:0'), 
        #                   torch.tensor([ -8.0552,   9.0451, -14.8539,   2.0358,  23.7501], device='cuda:0'), 
        #                   torch.tensor([-71.2067,  35.0843,  10.0034, -40.3933, 156.3643], device='cuda:0'), 
        #                   torch.tensor([ 64.3720,  29.1668,  25.8079, -54.8797, 149.2222], device='cuda:0'), 
        #                   torch.tensor([ 19.6085,  38.1664, -25.5673, -42.0857, 340.2368], device='cuda:0'), 
        #                   torch.tensor([ 69.5746,  20.8807, -29.1462, -21.2733, 331.4051], device='cuda:0'), 
        #                   torch.tensor([ 25.3657,  21.4144,  -0.9496, -14.2845, 251.9070], device='cuda:0'), 
        #                   torch.tensor([ 26.6365,  15.6355,   1.4626, -14.7403, 153.6139], device='cuda:0'), 
        #                   torch.tensor([ 69.5128,  12.9242,  15.4358,   7.2139, 231.6416], device='cuda:0'), 
        #                   torch.tensor([  6.4777,  34.7111,  18.6117, -50.6457, 204.0022], device='cuda:0'), 
        #                   torch.tensor([ 11.2704,  11.3726,  -9.5018, -33.7923,  96.3644], device='cuda:0'), 
        #                   torch.tensor([ 32.2507,  33.5692,  22.4197, -50.5921, 149.9413], device='cuda:0'), 
        #                   torch.tensor([  5.3813,  16.9567,  -0.8450, -16.7176, 192.4445], device='cuda:0'), 
        #                   torch.tensor([ 29.2941,   7.2327, -29.1579,  -5.4458, 270.0239], device='cuda:0'), 
        #                   torch.tensor([ 45.6147,  38.6352,  16.8563, -51.7608,  97.2458], device='cuda:0')]

        # 对selected poses的xyz坐标进行聚类
        pose_positions = torch.stack([pose[:3] for pose in selected_poses])  # (N, 3)
        print(f"总共{len(pose_positions)}个相机位置")

        # 使用KMeans聚类
        from sklearn.cluster import KMeans
        import numpy as np

        # 转为numpy进行聚类
        positions_np = pose_positions.cpu().numpy()
        n_clusters = min(8, len(positions_np))  # 最多8个聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(positions_np)

        print(f"KMeans聚类: {n_clusters}个聚类")

        # 计算每个聚类的平均位置
        cluster_avg_positions = []
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            if cluster_mask.any():
                cluster_positions = pose_positions[cluster_mask]
                avg_position = cluster_positions.mean(dim=0)
                cluster_avg_positions.append(avg_position)
                print(f"聚类{i}: {cluster_mask.sum()}个点, 平均位置: [{avg_position[0]:.1f}, {avg_position[1]:.1f}, {avg_position[2]:.1f}]")

        # 为每个聚类中心创建蓝色立方体
        all_cubes = []
        for center in cluster_avg_positions:
            cube_size = 2.0
            cube_density = 4
            offsets = torch.linspace(-cube_size/2, cube_size/2, cube_density, device=device)
            for dx in offsets:
                for dy in offsets:
                    for dz in offsets:
                        cube_point = center + torch.tensor([dx, dy, dz], device=device)
                        all_cubes.append(cube_point)

        # 组合可视化：GT点云 + 聚类中心蓝色立方体
        all_vis_points = torch.cat([gt_pc, torch.stack(all_cubes)], dim=0)
        cube_colors = torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(len(all_cubes), 1)  # 蓝色
        all_vis_colors = torch.cat([gt_colors, cube_colors], dim=0)

        vis = plot_point_cloud(all_vis_points, all_vis_colors,
                              name=f"KMeans Clustering: {n_clusters} clusters, GT+Blue Cluster Centers")
        vis.show()

        # TSP轨迹规划：从current pose到所有KMeans聚类中心
        current_pose = torch.cat([current_X_cam, current_V_cam], dim=0)
        current_location = current_pose[:3]  # 当前位置

        # 使用KMeans聚类中心进行TSP规划
        cluster_locations = torch.stack(cluster_avg_positions)  # 聚类中心位置
        all_locations = torch.cat([current_location.unsqueeze(0), cluster_locations], dim=0)  # 起点+聚类中心

        print(f"\nTSP规划: 起点 + {len(cluster_locations)}个聚类中心")
        print(f"起点位置: [{current_location[0]:.1f}, {current_location[1]:.1f}, {current_location[2]:.1f}]")

        # 计算距离矩阵（欧氏距离）
        n_points = len(all_locations)
        distance_matrix = torch.zeros(n_points, n_points, device=device)
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    distance_matrix[i, j] = torch.norm(all_locations[i] - all_locations[j])

        # 2-opt改进的TSP算法
        def solve_tsp_2opt(distance_matrix):
            n = distance_matrix.shape[0]

            # 初始解：贪心算法
            def greedy_tsp():
                visited = [False] * n
                path = [0]
                visited[0] = True
                current = 0

                for _ in range(n - 1):
                    next_node = -1
                    min_distance = float('inf')
                    for j in range(n):
                        if not visited[j] and distance_matrix[current, j] < min_distance:
                            min_distance = distance_matrix[current, j]
                            next_node = j
                    if next_node != -1:
                        visited[next_node] = True
                        path.append(next_node)
                        current = next_node
                return path

            # 计算路径总距离
            def calculate_distance(path):
                total = 0
                for i in range(len(path) - 1):
                    total += distance_matrix[path[i], path[i + 1]]
                return total

            # 2-opt交换
            def two_opt_swap(path, i, j):
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                return new_path

            # 获取初始解
            best_path = greedy_tsp()
            best_distance = calculate_distance(best_path)

            # 2-opt改进
            improved = True
            max_iterations = 100
            iteration = 0

            while improved and iteration < max_iterations:
                improved = False
                iteration += 1

                for i in range(1, len(best_path) - 1):
                    for j in range(i + 1, len(best_path)):
                        new_path = two_opt_swap(best_path, i, j)
                        new_distance = calculate_distance(new_path)

                        if new_distance < best_distance:
                            best_path = new_path
                            best_distance = new_distance
                            improved = True
                            break
                    if improved:
                        break

            print(f"2-opt优化: {iteration}次迭代")
            return best_path, best_distance

        # 求解TSP
        tsp_path, total_dist = solve_tsp_2opt(distance_matrix.cpu())
        ordered_locations = all_locations[tsp_path]

        print(f"TSP路径: {tsp_path}")
        print(f"总距离: {total_dist:.1f}")
        torch.tensor()

        # 生成轨迹线：在每两个位置之间插入点
        trajectory_points = []
        line_density = 10  # 每段线的点数

        for i in range(len(ordered_locations) - 1):
            start = ordered_locations[i]
            end = ordered_locations[i + 1]

            # 在start和end之间线性插值
            for t in torch.linspace(0, 1, line_density, device=device):
                point = start * (1 - t) + end * t
                trajectory_points.append(point)

        # 可视化组合
        vis_components = [gt_pc]  # GT点云
        vis_colors = [gt_colors]  # GT颜色

        # 起点黄色立方体
        start_cubes = []
        cube_size = 2.0
        cube_density = 4
        offsets = torch.linspace(-cube_size/2, cube_size/2, cube_density, device=device)
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    cube_point = current_location + torch.tensor([dx, dy, dz], device=device)
                    start_cubes.append(cube_point)

        vis_components.append(torch.stack(start_cubes))
        vis_colors.append(torch.tensor([[1.0, 1.0, 0.0]], device=device).repeat(len(start_cubes), 1))  # 黄色

        # 聚类中心蓝色立方体
        if all_cubes:
            vis_components.append(torch.stack(all_cubes))
            vis_colors.append(torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(len(all_cubes), 1))  # 蓝色

        # 轨迹线红色点
        if trajectory_points:
            vis_components.append(torch.stack(trajectory_points))
            vis_colors.append(torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(len(trajectory_points), 1))  # 红色

        # 最终可视化
        final_points = torch.cat(vis_components, dim=0)
        final_colors = torch.cat(vis_colors, dim=0)

        vis_tsp = plot_point_cloud(final_points, final_colors,
                                  name=f"TSP Trajectory: Yellow=Start, Blue=Clusters, Red=Path")
        vis_tsp.show()
        torch.tensor()

        if traj_index is None or traj_index == len(full_trajectory):
            # selected_pose_key = camera.select_top_n_collaborative_poses(
            #                                     mesh=mesh,
            #                                     mesh_for_check=mesh_for_check, 
            #                                     gt_point_cloud=gt_scene.return_entire_pt_cloud(return_features=False),
            #                                     n_poses=1000,
            #                                     depth_tolerance=1.0
            #                                 )
            t1 = time.time()
            final_pose_keys, all_pose_coverage_cache = camera.select_dense_graph_poses(
                                        mesh=mesh,
                                        mesh_for_check=mesh_for_check,
                                        gt_point_cloud=gt_scene.return_entire_pt_cloud(return_features=False),
                                        step_limit=1,  # 可调参数
                                        depth_tolerance=0.6
                                    )
            resulty = plan_path_with_mcts(final_pose_keys, all_pose_coverage_cache, '[6, 1, 5, 2, 5]', max_steps=100, simulation_count=300000, device=device)
            # resulty = resulty['mcts_result']
            print(len(resulty['anchor_path']))
            traj_index = 1
            full_trajectory = resulty['anchor_path']
            
            t2 = time.time()
            print(t2-t1)
        
        # gt_points = gt_scene.return_entire_pt_cloud(return_features=False)
        # gt_colors = torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(gt_points.shape[0], 1)

        # cube_size = 0.5 # 立方体大小，可调整
        # density = 10

        # cube_points_list = []
        # for point in splited_pose_space.values():
        #     # 提取坐标
        #     x, y, z = point
        #     ray_origin = point.cpu().numpy()
        #     ray_direction = np.array([0.0, 1.0, 0.0])
        #     intersections = mesh_for_check.ray.intersects_any(
        #         ray_origins=[ray_origin],
        #         ray_directions=[ray_direction]
        #     )
        #     if intersections[0]:
        #         continue
            
        #     # 在每个方向上生成点
        #     offsets = torch.linspace(-cube_size/2, cube_size/2, density, device=point.device)
            
        #     # 生成立方体内的所有点
        #     for dx in offsets:
        #         for dy in offsets:
        #             for dz in offsets:
        #                 cube_points_list.append([x + dx, y + dy, z + dz])

        # cube_points = torch.tensor(cube_points_list, device=device)

        # # 生成颜色（比如蓝色）
        # cube_colors = torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(cube_points.shape[0], 1)
        # fig = plot_point_cloud(torch.cat([gt_points, cube_points], dim=0), torch.cat([gt_colors, cube_colors], dim=0), name="100 anchor views")
        # fig.show()
        # torch.tensor()

        # # 移动到下一个位置（与原始代码完全一致）
        next_idx = torch.tensor(ast.literal_eval(full_trajectory[traj_index]), device=device)
        interpolation_step = 1
        for i in range(camera.n_interpolation_steps):
            camera.update_camera(next_idx, interpolation_step=interpolation_step)
            camera.capture_image(mesh)
            interpolation_step += 1
        traj_index += 1
        pose_i += resulty['anchor_distances'][traj_index-2]
    

    pc_plot = plot_point_cloud(full_pc, torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(full_pc.shape[0], 1), name='Filtered reconstructed surface points', 
                    point_size=2, max_points=150000, width=800, height=600, cmap='rgb')
    pc_plot.show()

    print("Trajectory computed in", time.time() - t0, "seconds.")
    print("Coverage Evolution:", coverage_evolution)
    return coverage_evolution, camera.X_cam_history, camera.V_cam_history

def run_tsp_test(params_name,
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
            
            # 如果是场景，缩放所有几何体
            mesh_for_check = trimesh.load(mesh_path)
            if isinstance(mesh_for_check, trimesh.Scene):
                # 使用 dump 方法合并场景
                mesh_for_check = mesh_for_check.dump(concatenate=True)

            mesh_for_check.vertices *= params.scene_scale_factor
            mesh_for_check = mesh_for_check.ray
            
            
            # mesh_for_check.vertices *= params.scene_scale_factor

            print("Mesh Vertices shape:", mesh.verts_list()[0].shape)
            print("Min Vert:", torch.min(mesh.verts_list()[0], dim=0)[0],
                  "\nMax Vert:", torch.max(mesh.verts_list()[0], dim=0)[0])

            # Use memory info to set frames and poses path
            scene_memory_path = os.path.join(scene_path, params.memory_dir_name)
            trajectory_nb = memory.current_epoch % memory.n_trajectories
            training_frames_path = memory.get_trajectory_frames_path(scene_memory_path, trajectory_nb)
            training_poses_path = memory.get_poses_path(scene_memory_path)

            torch.cuda.empty_cache()

            for start_cam_idx_i in range(len(settings.camera.start_positions)):
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

                coverage_evolution, X_cam_history, V_cam_history = compute_tsp_trajectory(params, macarons,
                                                                                      camera,
                                                                                      gt_scene, surface_scene,
                                                                                      proxy_scene, covered_scene,
                                                                                      mesh, mesh_for_check,
                                                                                      device,
                                                                                      settings)
                

                plot_scene_and_tragectory_and_constructed_pt(scene_name=scene_name, params=params, gt_scene=gt_scene,
                                                            proxy_scene=proxy_scene, macarons=macarons,
                                                            surface_scene=surface_scene, camera=camera,
                                                            i_th_scene=i_scene, memory=memory,
                                                            device=device, results_dir=results_dir)
                torch.tensor()

                dict_to_save[scene_name][str(start_cam_idx_i)] = {}
                dict_to_save[scene_name][str(start_cam_idx_i)]["coverage"] = coverage_evolution
                # dict_to_save[scene_name][str(start_cam_idx_i)]["X_cam_history"] = X_cam_history.cpu().numpy().tolist()
                # dict_to_save[scene_name][str(start_cam_idx_i)]["V_cam_history"] = V_cam_history.cpu().numpy().tolist()

                with open(results_json_path, 'w') as outfile:
                    json.dump(dict_to_save, outfile)
                print("Saved data about test losses in", results_json_name)

    print("All trajectories computed.")
