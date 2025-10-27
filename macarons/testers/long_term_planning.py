import os
import sys

import gc
import shutil
from ..utility.macarons_utils import *
from ..utility.utils import count_parameters
import json
import time
# from ..utility.diffusion_utils import *

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


    trained_weights = torch.load(model_path, map_location=device)
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


def compute_long_term_trajectory(params, macarons, camera, gt_scene, surface_scene, 
                           proxy_scene, covered_scene, mesh, device, settings, 
                           test_resolution=0.05, use_perfect_depth_map=False, 
                           compute_collision=False):

    macarons.eval()
    curriculum_distances = get_curriculum_sampling_distances(params, surface_scene, proxy_scene)
    full_pc = torch.zeros(0, 3, device=device)
    coverage_evolution = []
    t0 = time.time()

    def process_current_frame():
        """统一处理当前帧，修复X_cam参数"""
        try:
            current_frame = load_current_frame_perfect_depth(camera, device)
            depth, mask, error_mask, R, T = apply_perfect_depth_simple(current_frame, device)
            
            # 获取相机在世界坐标中的位置
            fov_camera = camera.get_fov_camera_from_RT(R_cam=R, T_cam=T)
            X_cam = fov_camera.get_camera_center()  # 🔥 正确获取相机位置
            
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
                'X_cam': X_cam,  # 🔥 正确传递相机位置
                'current_frame': current_frame
            }
            
        except FileNotFoundError as e:
            print(f"无法处理帧: {e}")
            return None

    for pose_i in range(params.n_poses_in_trajectory + 1):
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

        # 🔥 一次性处理当前帧
        frame_data = process_current_frame()
        if frame_data is None:
            continue

        # 更新场景（一次性完成）
        part_pc_features = torch.zeros(len(frame_data['part_pc']), 1, device=device)
        covered_scene.fill_cells(frame_data['part_pc'], features=part_pc_features)
        surface_scene.fill_cells(frame_data['part_pc'], features=part_pc_features)
        full_pc = torch.vstack((full_pc, frame_data['part_pc']))

        # 🔥 修复：正确更新代理点，传入正确的X_cam
        if frame_data['fov_proxy_mask'].any():
            fov_proxy_indices = proxy_scene.get_proxy_indices_from_mask(frame_data['fov_proxy_mask'])
            proxy_scene.fill_cells(frame_data['fov_proxy_points'], 
                                 features=fov_proxy_indices.view(-1, 1))
            
            # 🔥 关键修复：传入正确的相机位置
            proxy_scene.update_proxy_view_states(
                camera, frame_data['fov_proxy_mask'],
                signed_distances=frame_data['sgn_dists'],
                distance_to_surface=None, 
                X_cam=frame_data['X_cam']  # 🔥 使用正确的相机位置，而不是None
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

        if pose_i >= params.n_poses_in_trajectory:
            break

        # NBV选择逻辑保持不变
        with torch.no_grad():
            X_world, view_harmonics, occ_probs = compute_scene_occupancy_probability_field(
                params, macarons.scone, camera, surface_scene, proxy_scene, device
            )


        key_viewpoints = camera.sample_and_select_key_viewpoints(
                            mesh=mesh,
                            unknown_points=X_world[occ_probs.squeeze() > 0.5],  # 高概率occupancy点
                            covered_points=covered_scene.return_entire_pt_cloud(return_features=False),
                            num_samples=2000,  # 初始采样数
                            num_key_viewpoints=10,  # 选择10个关键视点
                            coverage_threshold=2 * test_resolution * params.scene_scale_factor  # 覆盖距离阈值
                        )

        print(key_viewpoints[0])
        torch.tensor()

        # 选择下一个最佳视点
        neighbor_indices = camera.get_neighboring_poses()
        valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)
        max_coverage_gain = -1.
        next_idx = valid_neighbors[0]

        for neighbor_i in range(len(valid_neighbors)):
            neighbor_idx = valid_neighbors[neighbor_i]
            neighbor_pose, _ = camera.get_pose_from_idx(neighbor_idx)
            X_neighbor, V_neighbor, fov_neighbor = camera.get_camera_parameters_from_pose(neighbor_pose)

            drop_neighbor = False
            if compute_collision:
                drop_neighbor = proxy_scene.camera_collides(params, camera, X_neighbor)

            if not drop_neighbor:
                with torch.no_grad():
                    _, _, visibility_gains, coverage_gain = predict_coverage_gain_for_single_camera(
                        params=params, macarons=macarons.scone,
                        proxy_scene=proxy_scene, surface_scene=surface_scene,
                        X_world=X_world, proxy_view_harmonics=view_harmonics, occ_probs=occ_probs,
                        camera=camera, X_cam_world=X_neighbor, fov_camera=fov_neighbor
                    )

                if coverage_gain.shape[0] > 0 and coverage_gain > max_coverage_gain:
                    max_coverage_gain = coverage_gain
                    next_idx = neighbor_idx

        # 移动到下一个位置
        interpolation_step = 1
        for i in range(camera.n_interpolation_steps):
            camera.update_camera(next_idx, interpolation_step=interpolation_step)
            camera.capture_image(mesh)
            interpolation_step += 1

    print("Trajectory computed in", time.time() - t0, "seconds.")
    print("Coverage Evolution:", coverage_evolution)
    return coverage_evolution, camera.X_cam_history, camera.V_cam_history
        
def run_long_term_test(params_name,
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

                coverage_evolution, X_cam_history, V_cam_history = compute_long_term_trajectory(params, macarons,
                                                                                      camera,
                                                                                      gt_scene, surface_scene,
                                                                                      proxy_scene, covered_scene,
                                                                                      mesh,
                                                                                      device,
                                                                                      settings,
                                                                                      test_resolution=test_resolution,
                                                                                      use_perfect_depth_map=use_perfect_depth_map,
                                                                                      compute_collision=compute_collision)
                

                # plot_scene_and_tragectory_and_constructed_pt(scene_name=scene_name, params=params, gt_scene=gt_scene,
                #                                             proxy_scene=proxy_scene, macarons=macarons,
                #                                             surface_scene=surface_scene, camera=camera,
                #                                             i_th_scene=i_scene, memory=memory,
                #                                             device=device, results_dir=results_dir)

                dict_to_save[scene_name][str(start_cam_idx_i)] = {}
                dict_to_save[scene_name][str(start_cam_idx_i)]["coverage"] = coverage_evolution
                # dict_to_save[scene_name][str(start_cam_idx_i)]["X_cam_history"] = X_cam_history.cpu().numpy().tolist()
                # dict_to_save[scene_name][str(start_cam_idx_i)]["V_cam_history"] = V_cam_history.cpu().numpy().tolist()

                with open(results_json_path, 'w') as outfile:
                    json.dump(dict_to_save, outfile)
                print("Saved data about test losses in", results_json_name)

    print("All trajectories computed.")
