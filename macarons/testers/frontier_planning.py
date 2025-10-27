import os
import sys
import ast
import gc
from ..utility.macarons_utils import *
from ..utility.utils import count_parameters
import json
import time
from ..utility.diffusion_utils import *
from ..utility.render_utils import plot_point_cloud
from ..utility.diffusion_utils import remove_close_points



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

    return gt_scene, covered_scene, surface_scene, proxy_scene, gt_surface, gt_normals


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


def compute_frontier_trajectory(params,
                       camera,
                       gt_scene, surface_scene, proxy_scene, covered_scene,
                       mesh, gt_surface, gt_normals,
                       device,
                       settings,
                       test_resolution=0.05,
                       use_perfect_depth_map=False,
                       compute_collision=False):

    full_pc = torch.zeros(0, 3, device=device)

    coverage_evolution = []
    t0 = time.time()

    current_best_pose = torch.zeros(5, device=device)
    path_record = 0
    Dijkstra_path = []

    visits_count = torch.zeros(gt_surface.size(0), dtype=gt_surface.dtype, device=device)

    splited_pose_space_idx = camera.generate_new_splited_dict()
    splited_pose_space = generate_key_value_splited_dict(splited_pose_space_idx)

    proxy_scene.proxy_supervision_occ = torch.full(
        (proxy_scene.n_proxy_points, 1),
        fill_value=-1,                 # -1 == 未知
        dtype=torch.int8,              # 或 int32，看需要
        device=device
        )
    

    for pose_i in range(params.n_poses_in_trajectory + 1):

        if pose_i % 10 == 0:
            print("Processing pose", str(pose_i) + "...")
        camera.fov_camera_0 = camera.fov_camera

        # if pose_i > 0 and pose_i % params.recompute_surface_every_n_loop == 0:
        #     print("Recomputing surface...")
        #     fill_surface_scene(surface_scene, full_pc,
        #                        random_sampling_max_size=params.n_gt_surface_points,
        #                        min_n_points_per_cell_fill=3,
        #                        progressive_fill=params.progressive_fill,
        #                        max_n_points_per_fill=params.max_points_per_progressive_fill)

        # ----------Predict visible surface points from RGB images------------------------------------------------------

        # Load input RGB image and camera pose
        all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(camera=camera,
                                                                                             n_frames=1,
                                                                                             n_alpha=params.n_alpha,
                                                                                             return_gt_zbuf=True)
        # Register GT surface points to compute true coverage for evaluation
        for i in range(all_zbuf[-1:].shape[0]):
            # TO CHANGE: filter points based on SSIM value!
            part_pc = camera.compute_partial_point_cloud(depth=all_zbuf[-1:],
                                                         mask=all_mask[-1:],
                                                         fov_cameras=camera.get_fov_camera_from_RT(
                                                             R_cam=all_R[-1:],
                                                             T_cam=all_T[-1:]),
                                                         gathering_factor=params.gathering_factor,
                                                         fov_range=params.sensor_range)

            # Fill surface scene
            part_pc_features = torch.zeros(len(part_pc), 1, device=device)
            covered_scene.fill_cells(part_pc, features=part_pc_features)  # Todo: change for better surface
        # Compute true coverage for evaluation
        current_coverage = gt_scene.scene_coverage(covered_scene,
                                                   surface_epsilon=2 * test_resolution * params.scene_scale_factor)
        if pose_i % 10 == 0:
            print("current coverage:", current_coverage)
        if current_coverage[0] == 0.:
            coverage_evolution.append(0.)
        else:
            coverage_evolution.append(current_coverage[0].item())

        if pose_i >= params.n_poses_in_trajectory:
            break


        # Format input as batches to feed depth model
        batch_dict, alpha_dict = create_batch_for_depth_model(params=params,
                                                              all_images=all_images, all_mask=all_mask,
                                                              all_R=all_R, all_T=all_T, all_zfar=all_zfar,
                                                              mode='inference', device=device,
                                                              all_zbuf=all_zbuf)

        # Depth prediction
        with torch.no_grad():
            depth, mask, error_mask, pose, gt_pose = apply_depth_model_imagination(params=params,
                                                                       batch_dict=batch_dict,
                                                                       alpha_dict=alpha_dict,
                                                                       device=device,
                                                                       use_perfect_depth=params.use_perfect_depth)

        if use_perfect_depth_map:
            depth = all_zbuf[2:3]
            error_mask = mask

        # We fill the surface scene with the partial point cloud
        for i in range(depth.shape[0]):
            # TO CHANGE: filter points based on SSIM value!
            part_pc = camera.compute_partial_point_cloud(depth=depth[i:i + 1],
                                                         mask=(mask * error_mask)[i:i + 1],
                                                         fov_cameras=camera.get_fov_camera_from_RT(
                                                             R_cam=batch_dict['R'][i:i + 1],
                                                             T_cam=batch_dict['T'][i:i + 1]),
                                                         gathering_factor=params.gathering_factor,
                                                         fov_range=params.sensor_range)

            # Fill surface scene
            part_pc_features = torch.zeros(len(part_pc), 1, device=device)
            # surface_scene.fill_cells(part_pc, features=part_pc_features)

            full_pc = torch.vstack((full_pc, part_pc))

        # ----------Update Proxy Points data with current FoV-----------------------------------------------------------

        # Get Proxy Points in current FoV
        fov_proxy_points, fov_proxy_mask = camera.get_points_in_fov(proxy_scene.proxy_points, return_mask=True,
                                                                    fov_camera=None,
                                                                    fov_range=params.sensor_range)
        fov_proxy_indices = proxy_scene.get_proxy_indices_from_mask(fov_proxy_mask)
        proxy_scene.fill_cells(fov_proxy_points, features=fov_proxy_indices.view(-1, 1))

        # Computing signed distance of proxy points in fov
        sgn_dists = camera.get_signed_distance_to_depth_maps(pts=fov_proxy_points,
                                                             depth_maps=depth,
                                                             mask=mask, fov_camera=None)

        proxy_scene.proxy_n_inside_fov[fov_proxy_mask] += 1
        
        # 2. 如果签名距离 >= -tol，则增加"在深度图后方"的计数
        proxy_scene.proxy_n_behind_depth[fov_proxy_mask] += (sgn_dists.view(-1, 1) >= -params.carving_tolerance).float()
        
        # 3. 计算空间点的占用概率，并根据阈值确定占用状态
        occupancy_probability = proxy_scene.proxy_n_behind_depth[fov_proxy_mask] / proxy_scene.proxy_n_inside_fov[fov_proxy_mask]
        
        # 4. 更新占用状态：
        # - 如果概率 >= 阈值，则为空(值为0)
        # - 如果概率 < 阈值，则为占用(值为1)
        proxy_scene.proxy_supervision_occ[fov_proxy_mask] = (occupancy_probability >= params.score_threshold).to(torch.int8)
        
        # 更新代理点的视野状态
        proxy_scene.update_proxy_out_of_field(fov_proxy_mask)
        
        # # Updating view_state vectors
        # proxy_scene.update_proxy_view_states(camera, fov_proxy_mask,
        #                                      signed_distances=sgn_dists,
        #                                      distance_to_surface=None, X_cam=None)  # distance_to_surface TO CHANGE!

        # # Update the supervision occupancy for proxy points using the signed distance
        # proxy_scene.update_proxy_supervision_occ(fov_proxy_mask, sgn_dists, tol=params.carving_tolerance)

        # # Update the out-of-field status for proxy points inside camera field of view
        # proxy_scene.update_proxy_out_of_field(fov_proxy_mask)

        # Update visibility history of surface points
        surface_scene.set_all_features_to_value(value=1.)

        # ----------Predict Occupancy Probability Field-----------------------------------------------------------------

        # with torch.no_grad():
        #     X_world, view_harmonics, occ_probs = compute_scene_occupancy_probability_field(params, macarons.scone,
        #                                                                                    camera,
        #                                                                                    surface_scene, proxy_scene,
        #                                                                                    device)

        # ----------Predict Coverage Gain of neighbor camera poses------------------------------------------------------


        occupied_mask = (proxy_scene.proxy_supervision_occ == 1).squeeze(-1)
        occupied_points = proxy_scene.proxy_points[occupied_mask]

        unknown_mask = (proxy_scene.proxy_supervision_occ == -1).squeeze(-1)
        unknown_points = proxy_scene.proxy_points[unknown_mask]

        # if (path_record + 1> len(Dijkstra_path)):
        #     sampled_poses_params_list = camera.sample_valid_poses_in_space_for_frontier(mesh, unknown_points, num_samples=1000)

        #     for best_sampled_pose in sampled_poses_params_list:
        #         if torch.equal(current_best_pose[:3], best_sampled_pose[0][:3]):
        #                 continue
        #         path_record = 0
        #         current_best_pose = best_sampled_pose[0]
        #         path_start_position = camera.cam_idx[:3].tolist()
        #         path_end_position = ast.literal_eval(best_sampled_pose[-2])[:3]
        #         Dijkstra_path = generate_Dijkstra_path(splited_pose_space, path_start_position, path_end_position, occupied_points, device)
                
        #         if Dijkstra_path: 
        #             break
        # Compute valid neighbor poses
        neighbor_indices = camera.get_neighboring_poses()
        valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)


        # filter_gt_points, mask_1 = remove_close_points(gt_surface, covered_scene.return_entire_pt_cloud(return_features=False), 2 * test_resolution * params.scene_scale_factor)
        # gt_normals_mask = gt_normals[mask_1]
        # counts_mask = visits_count[mask_1]
        # mask_for_valid_poses = (valid_neighbors[:, :3] == Dijkstra_path[path_record]).all(dim=1)
        # valid_candidate = valid_neighbors[mask_for_valid_poses]
        # next_idx = valid_candidate[torch.randint(len(valid_candidate), (), device=valid_candidate.device)]

        best_candiadate_count = -1.
        for neighbor_i in range(len(valid_neighbors)):
            neighbor_idx = valid_neighbors[neighbor_i]
            neighbor_pose, _ = camera.get_pose_from_idx(neighbor_idx)
            X_neighbor, V_neighbor, fov_neighbor = camera.get_camera_parameters_from_pose(neighbor_pose)
            fov_proxy_points, visits_points_mask, candiadate_count = camera.get_points_in_fov_vectorized(pts=gt_surface, normals=gt_normals, counts=visits_count, return_mask=True,
                                                            fov_camera=fov_neighbor, fov_range=1.1 * camera.zfar)

            if candiadate_count >= best_candiadate_count:
                best_candiadate_count = candiadate_count
                next_idx = neighbor_idx
        visits_count[visits_points_mask] += 1


        # ==============================================================================================================
        # Move to the neighbor NBV and acquire signal
        # ==============================================================================================================

        # Now that we have estimated the NBV among neighbors, we move toward this new camera pose and save RGB images
        # along the way.

        # ----------Move to next camera pose----------------------------------------------------------------------------
        # We move to the next pose and capture RGB images.
        interpolation_step = 1
        for i in range(camera.n_interpolation_steps):
            camera.update_camera(next_idx, interpolation_step=interpolation_step)
            camera.capture_image(mesh)
            interpolation_step += 1

        # ----------Depth prediction------------------------------------------------------------------------------------
        # Load input RGB image and camera pose
        all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(
            camera=camera,
            n_frames=params.n_interpolation_steps,
            n_alpha=params.n_alpha_for_supervision,
            return_gt_zbuf=True)

        # Format input as batches to feed depth model
        batch_dict, alpha_dict = create_batch_for_depth_model(params=params,
                                                              all_images=all_images, all_mask=all_mask,
                                                              all_R=all_R, all_T=all_T, all_zfar=all_zfar,
                                                              mode='supervision', device=device, all_zbuf=all_zbuf)

        # Depth prediction
        depth, mask, error_mask = [], [], []
        for i in range(batch_dict['images'].shape[0]):
            batch_dict_i = {}
            batch_dict_i['images'] = batch_dict['images'][i:i + 1]
            batch_dict_i['mask'] = batch_dict['mask'][i:i + 1]
            batch_dict_i['R'] = batch_dict['R'][i:i + 1]
            batch_dict_i['T'] = batch_dict['T'][i:i + 1]
            batch_dict_i['zfar'] = batch_dict['zfar'][i:i + 1]
            batch_dict_i['zbuf'] = batch_dict['zbuf'][i:i + 1]

            alpha_dict_i = {}
            alpha_dict_i['images'] = alpha_dict['images'][i:i + 1]
            alpha_dict_i['mask'] = alpha_dict['mask'][i:i + 1]
            alpha_dict_i['R'] = alpha_dict['R'][i:i + 1]
            alpha_dict_i['T'] = alpha_dict['T'][i:i + 1]
            alpha_dict_i['zfar'] = alpha_dict['zfar'][i:i + 1]
            alpha_dict_i['zbuf'] = alpha_dict['zbuf'][i:i + 1]

            with torch.no_grad():
                depth_i, mask_i, error_mask_i, _, _ = apply_depth_model_imagination(params=params,
                                                                        batch_dict=batch_dict_i,
                                                                        alpha_dict=alpha_dict_i,
                                                                        device=device,
                                                                        compute_loss=False,
                                                                        use_perfect_depth=params.use_perfect_depth)
                if use_perfect_depth_map:
                    depth_i = all_zbuf[2+i:3+i]
                    error_mask_i = mask_i

            depth.append(depth_i)
            mask.append(mask_i)
            error_mask.append(error_mask_i)
        depth = torch.cat(depth, dim=0)
        mask = torch.cat(mask, dim=0)
        error_mask = torch.cat(error_mask, dim=0)

        # ----------Build supervision signal from the new depth maps----------------------------------------------------
        all_part_pc = []
        all_fov_proxy_points = torch.zeros(0, 3, device=device)
        general_fov_proxy_mask = torch.zeros(params.n_proxy_points, device=device).bool()
        all_fov_proxy_mask = []
        all_sgn_dists = []
        all_X_cam = []
        all_fov_camera = []

        for i in range(depth.shape[0]):
            fov_frame = camera.get_fov_camera_from_RT(R_cam=batch_dict['R'][i:i + 1], T_cam=batch_dict['T'][i:i + 1])
            all_X_cam.append(fov_frame.get_camera_center())
            all_fov_camera.append(fov_frame)

            # TO CHANGE: filter points based on SSIM value!
            part_pc = camera.compute_partial_point_cloud(depth=depth[i:i + 1],
                                                         mask=(mask * error_mask)[i:i + 1].bool(),
                                                         fov_cameras=fov_frame,
                                                         gathering_factor=params.gathering_factor,
                                                         fov_range=params.sensor_range)

            # Surface points to fill surface scene
            all_part_pc.append(part_pc)

            # Get Proxy Points in current FoV
            fov_proxy_points, fov_proxy_mask = camera.get_points_in_fov(proxy_scene.proxy_points, return_mask=True,
                                                                        fov_camera=fov_frame, fov_range=params.sensor_range)
            all_fov_proxy_points = torch.vstack((all_fov_proxy_points, fov_proxy_points))
            all_fov_proxy_mask.append(fov_proxy_mask)
            general_fov_proxy_mask = general_fov_proxy_mask + fov_proxy_mask

            # Computing signed distance of proxy points in fov
            sgn_dists = camera.get_signed_distance_to_depth_maps(pts=fov_proxy_points, depth_maps=depth[i:i + 1],
                                                                 mask=mask[i:i + 1].bool(), fov_camera=fov_frame
                                                                 ).view(-1, 1)
            all_sgn_dists.append(sgn_dists)

            # Computing mask for proxy points close to the surface.


        # ----------Update Scenes to finalize supervision signal and prepare next iteration-----------------------------

        # 1. Surface scene
        # Fill surface scene
        # We give a visibility=1 to points that were visible in frame t, and 0 to others
        complete_part_pc = torch.vstack(all_part_pc)
        complete_part_pc_features = torch.zeros(len(complete_part_pc), 1, device=device)
        complete_part_pc_features[:len(all_part_pc[0])] = 1.
        surface_scene.fill_cells(complete_part_pc, features=complete_part_pc_features)

        full_pc = torch.vstack((full_pc, complete_part_pc))

        # Compute coverage gain for each new camera pose
        # We also include, at the beginning, the previous camera pose with a coverage gain equal to 0.
        # supervision_coverage_gains = torch.zeros(params.n_interpolation_steps, 1, device=device)
        # for i in range(depth.shape[0]):
        #     supervision_coverage_gains[i, 0] = surface_scene.camera_coverage_gain(all_part_pc[i],
        #                                                                           surface_epsilon=None)

        # Update visibility history of surface points
        surface_scene.set_all_features_to_value(value=1.)

        # 2. Proxy scene
        # Fill proxy scene
        general_fov_proxy_indices = proxy_scene.get_proxy_indices_from_mask(general_fov_proxy_mask)
        proxy_scene.fill_cells(proxy_scene.proxy_points[general_fov_proxy_mask],
                               features=general_fov_proxy_indices.view(-1, 1))

        # for i in range(depth.shape[0]):
        #     # Updating view_state vectors
        #     proxy_scene.update_proxy_view_states(camera, all_fov_proxy_mask[i],
        #                                          signed_distances=all_sgn_dists[i],
        #                                          distance_to_surface=None,
        #                                          X_cam=all_X_cam[i])  # distance_to_surface TO CHANGE!

        #     # Update the supervision occupancy for proxy points using the signed distance
        #     proxy_scene.update_proxy_supervision_occ(all_fov_proxy_mask[i], all_sgn_dists[i],
        #                                              tol=params.carving_tolerance)
        for i in range(depth.shape[0]):
            # 计算点的有符号距离
            proxy_scene.proxy_n_inside_fov[all_fov_proxy_mask[i]] += 1
            
            # 2. 如果签名距离 >= -tol，则增加"在深度图后方"的计数
            sgn_dists_i = all_sgn_dists[i]
            proxy_scene.proxy_n_behind_depth[all_fov_proxy_mask[i]] += (sgn_dists_i >= -params.carving_tolerance).float()
            
            # 3. 计算空间点的占用概率，并根据阈值确定占用状态
            occupancy_probability = proxy_scene.proxy_n_behind_depth[all_fov_proxy_mask[i]] / proxy_scene.proxy_n_inside_fov[all_fov_proxy_mask[i]]
            
            # 4. 更新占用状态：
            # - 如果概率 >= 阈值，则为空(值为0)
            # - 如果概率 < 阈值，则为占用(值为1)
            proxy_scene.proxy_supervision_occ[all_fov_proxy_mask[i]] = (occupancy_probability > params.score_threshold).to(torch.int8)


        # Update the out-of-field status for proxy points inside camera field of view
        proxy_scene.update_proxy_out_of_field(general_fov_proxy_mask)
        path_record += 1
        # if pose_i == 10:
        #     # --- 处理占用点 ---
        #     occupied_mask = (proxy_scene.proxy_supervision_occ == 1).squeeze(-1)
        #     occupied_points = proxy_scene.proxy_points[occupied_mask]
        #     n_occupied = occupied_points.shape[0]
        #     occupied_colors = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)  # 绿色 
        #     occupied_colors_repeated = occupied_colors.repeat(n_occupied, 1)

        #     # --- 处理空点 ---
        #     empty_mask = (proxy_scene.proxy_supervision_occ == 0).squeeze(-1)
        #     empty_points = proxy_scene.proxy_points[empty_mask]
        #     n_empty = empty_points.shape[0]
        #     empty_colors = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)  # 红色
        #     empty_colors_repeated = empty_colors.repeat(n_empty, 1)
            
        #     # --- 处理未知点 ---
        #     unknown_mask = (proxy_scene.proxy_supervision_occ == -1).squeeze(-1)
        #     unknown_points = proxy_scene.proxy_points[unknown_mask]
        #     n_unknown = unknown_points.shape[0]
        #     unknown_colors = torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32, device=device)  # 灰色
        #     unknown_colors_repeated = unknown_colors.repeat(n_unknown, 1)

        #     # --- 处理全点云 ---
        #     n_full = full_pc.shape[0]
        #     full_pc_colors = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)  # 蓝色
        #     full_pc_colors_repeated = full_pc_colors.repeat(n_full, 1)

        #     # --- 合并所有点云 ---
        #     combined_points = torch.cat([occupied_points, empty_points, unknown_points, full_pc], dim=0)
        #     combined_colors = torch.cat([
        #         occupied_colors_repeated, 
        #         empty_colors_repeated, 
        #         unknown_colors_repeated, 
        #         full_pc_colors_repeated
        #     ], dim=0)

        #     # --- 绘制合并后的点云 ---
        #     fig_combined = plot_point_cloud(combined_points, combined_colors)
        #     fig_combined.show()


    
    print("Trajectory computed in", time.time() - t0, "seconds.")
    # blender_X, blender_look = create_blender_curves(params, camera.X_cam_history, camera.V_cam_history,
    #   mirrored_pose=False)
    print("Coverage Evolution:", coverage_evolution)

    return coverage_evolution, camera.X_cam_history, camera.V_cam_history


def run_frontier_planning(params_name,
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
                gt_scene, covered_scene, surface_scene, proxy_scene, gt_surface, gt_normals  = setup_test_scene(params,
                                                                                       mesh,
                                                                                       settings,
                                                                                       mirrored_scene,
                                                                                       device,
                                                                                       mirrored_axis=mirrored_axis,
                                                                                       test_resolution=test_resolution)

                camera = setup_test_camera(params, mesh, start_cam_idx, settings, occupied_pose_data,
                                           device, training_frames_path,
                                           mirrored_scene=mirrored_scene, mirrored_axis=mirrored_axis)
                print(camera.X_cam_history[0], camera.V_cam_history[0])

                coverage_evolution, X_cam_history, V_cam_history = compute_frontier_trajectory(params,
                                                                                      camera,
                                                                                      gt_scene, surface_scene,
                                                                                      proxy_scene, covered_scene,
                                                                                      mesh, gt_surface, gt_normals,
                                                                                      device,
                                                                                      settings,
                                                                                      test_resolution=test_resolution,
                                                                                      use_perfect_depth_map=use_perfect_depth_map,
                                                                                      compute_collision=compute_collision)
                
                plot_scene_and_tragectory_and_constructed_pt(scene_name=scene_name, params=params, gt_scene=gt_scene,
                                                            proxy_scene=proxy_scene, macarons=macarons,
                                                            surface_scene=surface_scene, camera=camera,
                                                            i_th_scene=i_scene, memory=memory,
                                                            device=device, results_dir=results_dir)

                dict_to_save[scene_name][str(start_cam_idx_i)] = {}
                dict_to_save[scene_name][str(start_cam_idx_i)]["coverage"] = coverage_evolution
                # dict_to_save[scene_name][str(start_cam_idx_i)]["X_cam_history"] = X_cam_history.cpu().numpy().tolist()
                # dict_to_save[scene_name][str(start_cam_idx_i)]["V_cam_history"] = V_cam_history.cpu().numpy().tolist()

                with open(results_json_path, 'w') as outfile:
                    json.dump(dict_to_save, outfile)
                print("Saved data about test losses in", results_json_name)

    print("All trajectories computed.")
