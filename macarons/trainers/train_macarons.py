import sys
import json
import time

from ..utility.macarons_utils import *
from ..utility.utils import count_parameters, check_gradients

dir_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(dir_path, "../../data/scenes")
weights_dir = os.path.join(dir_path, "../../weights/macarons")


def setup_scene(params,
                mesh,
                settings,
                mirrored_scene,
                device, is_master,
                mirrored_axis=None):
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
                     cell_resolution=None,
                     n_proxy_points=params.n_proxy_points,
                     device=device,
                     view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                     feature_dim=3,
                     mirrored_scene=mirrored_scene,
                     mirrored_axis=mirrored_axis)  # We use colors as features

    # We fill gt_scene with points sampled on the surface of the ground truth mesh
    gt_surface = get_scene_gt_surface(gt_scene=gt_scene,
                                      verts=mesh.verts_list()[0],
                                      faces=mesh.faces_list()[0],
                                      n_surface_points=params.n_gt_surface_points)
    gt_scene.fill_cells(gt_surface)

    # Initialize surface_scene: we store in this scene the surface points computed by the depth module from RGB images
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
                          feature_dim=1,  # We use visibility history as features
                          mirrored_scene=mirrored_scene,
                          mirrored_axis=mirrored_axis)

    # Initialize proxy_scene: we store in this scene the proxy points, which are used to encode the volumetric occupancy
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
    print("Proxy scene initialized with score_threshold =", proxy_scene.score_threshold)

    return gt_scene, surface_scene, proxy_scene


def setup_camera(params,
                 mesh, proxy_scene,
                 settings,
                 occupied_pose_data,
                 device, is_master,
                 training_frames_path,
                 mirrored_scene,
                 mirrored_axis=None):
    """
    Setup the camera used for prediction.

    :param params:
    :param mesh:
    :param settings:
    :param occupied_pose_data:
    :param device:
    :param is_master:
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
                                device=device)

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

    # Select a random, valid camera pose as starting pose
    start_cam_idx = camera.get_random_valid_pose(mesh=mesh, proxy_scene=proxy_scene)
    camera.initialize_camera(start_cam_idx=start_cam_idx)

    # Capture initial image
    camera.capture_image(mesh)

    # Move to a valid neighbor pose before starting training.
    # Thus, we will have a few images to start training the depth module
    neighbor_indices = camera.get_neighboring_poses()
    valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)
    next_idx = valid_neighbors[np.random.randint(low=0, high=len(valid_neighbors))]

    # We capture images along the way
    interpolation_step = 1
    for i in range(camera.n_interpolation_steps):
        camera.update_camera(next_idx, interpolation_step=interpolation_step)
        camera.capture_image(mesh)
        interpolation_step += 1

    return camera


def loop(params, batch, mesh,
         camera, surface_scene, proxy_scene, surface_distance, n_cell_per_occ_forward_pass,
         macarons, freeze,
         depth_loss_fn, ssim_loss_fn, regularity_loss_fn, pose_loss_fn, occ_loss_fn, cov_loss_fn,
         device, is_master,
         full_pc,
         loop_time=None,
         warmup_phase=False
         ):
    """

    :param params:
    :param batch:
    :param mesh: (Mesh)
    :param camera: (Camera)
    :param gt_scene: (Scene)
    :param surface_scene: (Scene)
    :param proxy_scene: (Scene)
    :param macarons: (Macarons)
    :param depth_loss_fn:
    :param occ_loss_fn:
    :param cov_loss_fn:
    :param device:
    :param is_master:
    :param full_pc: (Tensor) Total surface point cloud tensor with shape (N, 3).
    :param loop_time: (list)
    :return:
    """

    mesh_verts = mesh.verts_list()[0]

    # ==================================================================================================================
    # Path Planning
    # ==================================================================================================================

    # Initial prediction is performed with no gradient.
    # Its purpose is to select the next best pose among neighbors for active path planning.

    macarons.eval()
    # Set current camera to prediction camera
    camera.fov_camera_0 = camera.fov_camera

    if loop_time is not None:
        decision_making_start_time = time.time()

    # ----------Predict visible surface points from RGB images----------------------------------------------------------

    # Load input RGB image and camera pose
    if not params.use_perfect_depth:
        all_images, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(camera=camera,
                                                                                   n_frames=1,
                                                                                   n_alpha=params.n_alpha)
        all_zbuf = None
    else:
        all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(camera=camera,
                                                                                             n_frames=1,
                                                                                             n_alpha=params.n_alpha,
                                                                                             return_gt_zbuf=True)

    # Format input as batches to feed depth model
    batch_dict, alpha_dict = create_batch_for_depth_model(params=params,
                                                          all_images=all_images, all_mask=all_mask,
                                                          all_R=all_R, all_T=all_T, all_zfar=all_zfar,
                                                          mode='inference', device=device,
                                                          all_zbuf=all_zbuf)

    # Depth prediction
    with torch.no_grad():
        depth, mask, error_mask, pose, gt_pose = apply_depth_model(params=params,
                                                                   macarons=macarons.depth,
                                                                   batch_dict=batch_dict,
                                                                   alpha_dict=alpha_dict,
                                                                   device=device,
                                                                   use_perfect_depth=params.use_perfect_depth)

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
        surface_scene.fill_cells(part_pc, features=part_pc_features)

        full_pc[0] = torch.vstack((full_pc[0], part_pc))

    # ----------Update Proxy Points data with current FoV---------------------------------------------------------------

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

    # Updating view_state vectors
    proxy_scene.update_proxy_view_states(camera, fov_proxy_mask,
                                         signed_distances=sgn_dists,
                                         distance_to_surface=None, X_cam=None)  # distance_to_surface TO CHANGE!

    # Update the supervision occupancy for proxy points using the signed distance
    proxy_scene.update_proxy_supervision_occ(fov_proxy_mask, sgn_dists, tol=params.carving_tolerance)

    # Update the out-of-field status for proxy points inside camera field of view
    proxy_scene.update_proxy_out_of_field(fov_proxy_mask)

    # Update visibility history of surface points
    surface_scene.set_all_features_to_value(value=1.)

    # ----------Predict Occupancy Probability Field---------------------------------------------------------------------

    with torch.no_grad():
        X_world, view_harmonics, occ_probs = compute_scene_occupancy_probability_field(params, macarons.scone, camera,
                                                                                       surface_scene, proxy_scene,
                                                                                       device,
                                                                                       use_supervision_occ_instead_of_predicted=warmup_phase)

    # ----------Predict Coverage Gain of neighbor camera poses----------------------------------------------------------

    # Compute valid neighbor poses
    neighbor_indices = camera.get_neighboring_poses()
    valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)

    max_coverage_gain = -1.
    next_idx = valid_neighbors[0]

    # For each valid neighbor...
    for neighbor_i in range(len(valid_neighbors)):
        neighbor_idx = valid_neighbors[neighbor_i]
        neighbor_pose, _ = camera.get_pose_from_idx(neighbor_idx)
        X_neighbor, V_neighbor, fov_neighbor = camera.get_camera_parameters_from_pose(neighbor_pose)

        # ...We predict its coverage gain...
        with torch.no_grad():
            _, _, visibility_gains, coverage_gain = predict_coverage_gain_for_single_camera(
                params=params, macarons=macarons.scone,
                proxy_scene=proxy_scene, surface_scene=surface_scene,
                X_world=X_world, proxy_view_harmonics=view_harmonics, occ_probs=occ_probs,
                camera=camera, X_cam_world=X_neighbor, fov_camera=fov_neighbor)

        # ...And save it with the neighbor index if the estimated coverage is better
        if coverage_gain.shape[0] > 0 and coverage_gain > max_coverage_gain:
            max_coverage_gain = coverage_gain
            next_idx = neighbor_idx

    X_cam_t = 0. + camera.X_cam
    V_cam_t = 0. + camera.V_cam
    fov_camera_t = camera.get_fov_camera_from_XV(X_cam=X_cam_t, V_cam=V_cam_t)

    if loop_time is not None:
        decision_making_end_time = time.time()

    # ==================================================================================================================
    # Move to the neighbor NBV and acquire supervision signal
    # ==================================================================================================================

    # Now that we have estimated the NBV among neighbors, we move toward this new camera pose and save RGB images along
    # the way.
    # We will use these images to generate a supervision signal for the the entire model, and compare this signal to the
    # prediction made by the model from the current state.

    if params.online_learning:
        macarons.train()

    # ----------Move to next camera pose--------------------------------------------------------------------------------
    # We move to the next pose and capture RGB images.
    interpolation_step = 1
    for i in range(camera.n_interpolation_steps):
        camera.update_camera(next_idx, interpolation_step=interpolation_step)
        camera.capture_image(mesh)
        interpolation_step += 1

    # ----------Depth prediction----------------------------------------------------------------------------------------
    if loop_time is not None:
        memory_building_start_time = time.time()

    # Load input RGB image and camera pose
    if not params.use_perfect_depth:
        all_images, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(camera=camera,
                                                                               n_frames=params.n_interpolation_steps,
                                                                               n_alpha=params.n_alpha_for_supervision)
        all_zbuf = None
    else:
        all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(camera=camera,
                                                                               n_frames=params.n_interpolation_steps,
                                                                               n_alpha=params.n_alpha_for_supervision,
                                                                               return_gt_zbuf=True)
    # Format input as batches to feed depth model
    batch_dict, alpha_dict = create_batch_for_depth_model(params=params,
                                                          all_images=all_images, all_mask=all_mask,
                                                          all_R=all_R, all_T=all_T, all_zfar=all_zfar,
                                                          mode='supervision', device=device, all_zbuf=all_zbuf)
    # Depth prediction and depth loss computation
    depth_loss, depth, mask, error_mask, pose, gt_pose = apply_depth_model(params=params, macarons=macarons.depth,
                                                                           batch_dict=batch_dict, alpha_dict=alpha_dict,
                                                                           device=device,
                                                                           depth_loss_fn=depth_loss_fn,
                                                                           pose_loss_fn=pose_loss_fn,
                                                                           regularity_loss_fn=regularity_loss_fn,
                                                                           ssim_loss_fn=ssim_loss_fn,
                                                                           compute_loss=True,
                                                                           use_perfect_depth=params.use_perfect_depth)

    # ----------Build supervision signal from the new depth maps--------------------------------------------------------
    all_part_pc = []
    all_fov_proxy_points = torch.zeros(0, 3, device=device)
    general_fov_proxy_mask = torch.zeros(params.n_proxy_points, device=device).bool()
    all_fov_proxy_mask = []
    all_sgn_dists = []
    all_X_cam = []
    all_fov_camera = []

    close_fov_proxy_mask = torch.zeros(params.n_proxy_points, device=device).bool()

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

        # Computing mask for proxy points close to the surface. We will use this for occupancy probability supervision.
        close_fov_proxy_mask[fov_proxy_mask] = False + (sgn_dists.view(-1).abs() < surface_distance)

    # ----------Use first info about supervision signals to sample points for prediction--------------------------------

    if params.online_learning and (not freeze):
        # with macarons.no_sync():
        # Predicts occupancy probability and updates them in proxy_scene
        close_fov_proxy_mask = close_fov_proxy_mask * (proxy_scene.out_of_field[..., 0] < 1.)  # todo: useless?
        prediction_mask, predicted_occs = compute_occupancy_probability_for_supervision(params, macarons.scone,
                                                                                        camera, proxy_scene,
                                                                                        close_fov_proxy_mask,
                                                                                        surface_scene,
                                                                                        n_cell_per_occ_forward_pass,
                                                                                        device)

        # Predicts coverage gain
        # We also predict coverage gain for the previous camera. We expect the coverage gain to be 0.
        predicted_coverage_gains = torch.zeros(params.n_interpolation_steps, 1, device=device)
        for i in range(depth.shape[0]):
            X_cam = all_X_cam[i]
            fov_camera = all_fov_camera[i]

            # ...We predict its coverage gain...
            _, _, visibility_gains, predicted_coverage_gain = predict_coverage_gain_for_single_camera(
                params=params, macarons=macarons.scone,
                proxy_scene=proxy_scene, surface_scene=surface_scene,
                X_world=X_world, proxy_view_harmonics=view_harmonics, occ_probs=occ_probs,
                camera=camera, X_cam_world=X_cam, fov_camera=fov_camera)
            if len(predicted_coverage_gain) > 0:
                predicted_coverage_gains[i:i+1] += predicted_coverage_gain

    # ----------Update Scenes to finalize supervision signal and prepare next iteration---------------------------------

    # 1. Surface scene
    # Fill surface scene
    # We give a visibility=1 to points that were visible in frame t, and 0 to others
    complete_part_pc = torch.vstack(all_part_pc)
    complete_part_pc_features = torch.zeros(len(complete_part_pc), 1, device=device)
    complete_part_pc_features[:len(all_part_pc[0])] = 1.
    surface_scene.fill_cells(complete_part_pc, features=complete_part_pc_features)

    full_pc[0] = torch.vstack((full_pc[0], complete_part_pc))

    # Compute coverage gain for each new camera pose
    # We also include, at the beginning, the previous camera pose with a coverage gain equal to 0.
    supervision_coverage_gains = torch.zeros(params.n_interpolation_steps, 1, device=device)
    if params.online_learning:
        for i in range(depth.shape[0]):
            supervision_coverage_gains[i, 0] = surface_scene.camera_coverage_gain(all_part_pc[i],
                                                                                  surface_epsilon=None,
                                                                                  surface_epsilon_factor=params.surface_epsilon_factor)

    # Update visibility history of surface points
    surface_scene.set_all_features_to_value(value=1.)

    # 2. Proxy scene
    # Fill proxy scene
    general_fov_proxy_indices = proxy_scene.get_proxy_indices_from_mask(general_fov_proxy_mask)
    proxy_scene.fill_cells(proxy_scene.proxy_points[general_fov_proxy_mask],
                           features=general_fov_proxy_indices.view(-1, 1))

    for i in range(depth.shape[0]):
        # Updating view_state vectors
        proxy_scene.update_proxy_view_states(camera, all_fov_proxy_mask[i],
                                             signed_distances=all_sgn_dists[i],
                                             distance_to_surface=None,
                                             X_cam=all_X_cam[i])  # distance_to_surface TO CHANGE!

        # Update the supervision occupancy for proxy points using the signed distance
        proxy_scene.update_proxy_supervision_occ(all_fov_proxy_mask[i], all_sgn_dists[i], tol=params.carving_tolerance)

    # Update the out-of-field status for proxy points inside camera field of view
    proxy_scene.update_proxy_out_of_field(general_fov_proxy_mask)

    # ==================================================================================================================
    # Compute Loss
    # ==================================================================================================================
    # predicted_occs = proxy_scene.proxy_proba[prediction_mask]

    # Occupancy Loss
    if params.online_learning and (not freeze) and (len(predicted_occs)) > 0:
        occ_loss_computed = True
        supervision_occs = proxy_scene.proxy_supervision_occ[prediction_mask]
        occ_loss = occ_loss_fn(predicted_occs, supervision_occs) \
                   * predicted_occs.shape[0] / params.n_proxy_point_for_occupancy_supervision
    else:
        occ_loss_computed = False
        occ_loss = torch.zeros(1, device=device)[0]

    # Coverage Loss
    if params.online_learning and (not freeze) and (len(predicted_coverage_gains) > 0):
        cov_loss_computed = True
        cov_loss = cov_loss_fn(predicted_coverage_gains.view(1, -1, 1), supervision_coverage_gains.view(1, -1, 1))
    else:
        cov_loss_computed = False
        cov_loss = torch.zeros(1, device=device)[0]

    # Total loss
    scone_loss = occ_loss + cov_loss

    if loop_time is not None:
        memory_building_end_time = time.time()

    if batch % params.empty_cache_every_n_batch == 0:  # and is_master:
        with torch.no_grad():
            if is_master:
                print("Depth loss:", depth_loss)
                print("Occupancy loss:", occ_loss)
                print("Coverage loss:", cov_loss)
                print("Depth norm:", to_python_float(torch.linalg.norm((depth*mask.bool()).detach())))

                if params.online_learning and (not freeze):
                    print("Predicted occs sum:", predicted_occs.detach().abs().sum())
                    print("Supervision occs sum:", supervision_occs.detach().sum())
                    print("supervision occs shape:", supervision_occs.shape)
                    print("Predicted cov:", predicted_coverage_gains.detach())
                    print("Supervision cov:", supervision_coverage_gains.detach())
                    print("Visibility gains min/max:", visibility_gains.detach().min(), visibility_gains.detach().max())

    if occ_loss_computed:
        occ_loss = occ_loss.detach()
    if cov_loss_computed:
        cov_loss = cov_loss.detach()

    if loop_time is not None:
        loop_time.append(decision_making_end_time - decision_making_start_time
                         + memory_building_end_time - memory_building_start_time)

    return scone_loss, depth_loss, occ_loss, cov_loss

    # return loss, depth_loss.detach(), occ_loss.detach(), cov_loss.detach()  # , \
    # predicted_occs.detach(), supervision_occs, \
    # predicted_coverage_gains.detach(), supervision_coverage_gains


def memory_loop(params, batch, camera, memory,
                macarons, depth_loss_fn, ssim_loss_fn, regularity_loss_fn, pose_loss_fn,
                device, is_master):
    batch_dict, alpha_dict = memory.get_random_batch_for_depth_model(params, camera,
                                                                     n_sample=params.n_memory_samples,
                                                                     alphas=params.alphas,
                                                                     mode='supervision')

    depth_loss, depth, mask, _, _, _ = apply_depth_model(params=params, macarons=macarons.depth,
                                                         batch_dict=batch_dict, alpha_dict=alpha_dict,
                                                         device=device,
                                                         depth_loss_fn=depth_loss_fn,
                                                         pose_loss_fn=pose_loss_fn,
                                                         regularity_loss_fn=regularity_loss_fn,
                                                         ssim_loss_fn=ssim_loss_fn,
                                                         compute_loss=True)

    if batch % params.empty_cache_every_n_batch == 0:  # and is_master:
        with torch.no_grad():
            if is_master:
                print("Depth loss:", depth_loss)
                print("Depth norm:", to_python_float(torch.linalg.norm((depth*mask.bool()).detach())))

    return depth_loss


# if freeze, don't call this method!
def memory_scene_loop(params, batch, memory, camera, depths_memory_path,
                      partial_surface_scene, total_surface_scene, proxy_scene, pseudo_gt_proxy_proba,
                      prediction_camera, current_depth_i,
                      surface_distance, n_cell_per_occ_forward_pass,
                      macarons, occ_loss_fn, cov_loss_fn,
                      device, is_master, print_result=False,
                      supervise_with_online_field=False,
                      warmup_phase=False,
                      depth_list=None):

    # memory_scene_dict = memory.get_random_scene_for_scone_model(params, camera, device, n_weights_update)
    # partial_surface_scene = memory_scene_dict['partial_surface_scene'],
    # total_surface_scene = memory_scene_dict['total_surface_scene']
    # proxy_scene = memory_scene_dict['proxy_scene']
    # prediction_camera = memory_scene_dict['prediction_camera']
    # traj_depth_nb = memory_scene_dict['traj_depth_nb']
    # start_depth_i = memory_scene_dict['start_depth_i']

    # ----------Predict Occupancy Probability Field---------------------------------------------------------------------

    with torch.no_grad():
        X_world, view_harmonics, occ_probs = compute_scene_occupancy_probability_field(params, macarons.scone, None,
                                                                                       partial_surface_scene,
                                                                                       proxy_scene,
                                                                                       device,
                                                                                       prediction_camera=prediction_camera,
                                                                                       use_supervision_occ_instead_of_predicted=warmup_phase)

    # ----------Build supervision signal from the new depth maps--------------------------------------------------------
    all_part_pc = []
    all_fov_proxy_points = torch.zeros(0, 3, device=device)
    general_fov_proxy_mask = torch.zeros(params.n_proxy_points, device=device).bool()
    all_fov_proxy_mask = []
    all_sgn_dists = []
    all_X_cam = []
    all_fov_camera = []

    close_fov_proxy_mask = torch.zeros(params.n_proxy_points, device=device).bool()

    for i in range(params.n_poses_in_memory_scene_loops):
        if i == 0:
            depth_i = depth_list[-1]
        else:
            if params.random_poses_in_memory_scene_loops:
                n_total_depths = len(os.listdir(depths_memory_path))
                tmp_mask = torch.ones(n_total_depths).bool()
                tmp_mask[torch.Tensor(depth_list).long()] = False
                depth_i = int(np.random.choice(torch.arange(start=0, end=n_total_depths)[tmp_mask].numpy()))
            else:
                depth_i = current_depth_i + i
            depth_list.append(depth_i)
        depth_dict = torch.load(os.path.join(depths_memory_path, str(depth_i) + '.pt'), map_location=device)
        depth = depth_dict['depth']
        mask = depth_dict['mask']
        error_mask = depth_dict['error_mask']

        fov_frame = camera.get_fov_camera_from_RT(R_cam=depth_dict['R'][0:0 + 1],
                                                  T_cam=depth_dict['T'][0:0 + 1])
        all_X_cam.append(fov_frame.get_camera_center())
        all_fov_camera.append(fov_frame)

        # TO CHANGE: filter points based on SSIM value!
        part_pc = camera.compute_partial_point_cloud(depth=depth[0:0 + 1],
                                                     mask=(mask * error_mask)[0:0 + 1].bool(),
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
        sgn_dists = camera.get_signed_distance_to_depth_maps(pts=fov_proxy_points, depth_maps=depth[0:0 + 1],
                                                             mask=mask[0:0 + 1].bool(), fov_camera=fov_frame
                                                             ).view(-1, 1)
        all_sgn_dists.append(sgn_dists)

        # Computing mask for proxy points close to the surface. We will use this for occupancy probability supervision.
        close_fov_proxy_mask[fov_proxy_mask] = False + (sgn_dists.view(-1).abs() < surface_distance)

    # ----------Use first info about supervision signals to sample points for prediction--------------------------------

    # with macarons.no_sync():
    # Predicts occupancy probability and updates them in proxy_scene
    close_fov_proxy_mask = close_fov_proxy_mask * (proxy_scene.out_of_field[..., 0] < 1.)
    prediction_mask, predicted_occs = compute_occupancy_probability_for_supervision(params, macarons.scone,
                                                                                    None, proxy_scene,
                                                                                    close_fov_proxy_mask,
                                                                                    partial_surface_scene,
                                                                                    n_cell_per_occ_forward_pass,
                                                                                    device,
                                                                                    prediction_camera=prediction_camera)

    # Predicts coverage gain
    # We also predict coverage gain for the previous camera. We expect the coverage gain to be 0.
    predicted_coverage_gains = torch.zeros(params.n_poses_in_memory_scene_loops, 1, device=device)
    for i in range(params.n_poses_in_memory_scene_loops):
        X_cam = all_X_cam[i]
        fov_camera = all_fov_camera[i]

        # ...We predict its coverage gain...
        _, _, visibility_gains, predicted_coverage_gain = predict_coverage_gain_for_single_camera(
            params=params, macarons=macarons.scone,
            proxy_scene=proxy_scene, surface_scene=partial_surface_scene,
            X_world=X_world, proxy_view_harmonics=view_harmonics, occ_probs=occ_probs,
            camera=camera, X_cam_world=X_cam, fov_camera=fov_camera, prediction_camera=prediction_camera)
        if len(predicted_coverage_gain) > 0:
            predicted_coverage_gains[i:i + 1] += predicted_coverage_gain

    # ----------Update Scenes to finalize supervision signal and prepare next iteration---------------------------------

    # 1. Surface scene
    # Fill surface scene
    # We give a visibility=1 to points that were visible in frame t, and 0 to others
    complete_part_pc = torch.vstack(all_part_pc)
    complete_part_pc_features = torch.zeros(len(complete_part_pc), 1, device=device)
    complete_part_pc_features[:len(all_part_pc[0])] = 1.
    partial_surface_scene.fill_cells(complete_part_pc, features=complete_part_pc_features)

    # Compute coverage gain for each new camera pose
    # We also include, at the beginning, the previous camera pose with a coverage gain equal to 0.
    supervision_coverage_gains = torch.zeros(params.n_poses_in_memory_scene_loops, 1, device=device)
    for i in range(params.n_poses_in_memory_scene_loops):
        supervision_coverage_gains[i, 0] = partial_surface_scene.camera_coverage_gain(all_part_pc[i],
                                                                                      surface_epsilon=None,
                                                                                      surface_epsilon_factor=params.surface_epsilon_factor)

    # Update visibility history of surface points
    partial_surface_scene.set_all_features_to_value(value=1.)

    # 2. Proxy scene
    # Fill proxy scene
    general_fov_proxy_indices = proxy_scene.get_proxy_indices_from_mask(general_fov_proxy_mask)
    proxy_scene.fill_cells(proxy_scene.proxy_points[general_fov_proxy_mask],
                           features=general_fov_proxy_indices.view(-1, 1))

    for i in range(params.n_poses_in_memory_scene_loops):
        # Updating view_state vectors
        proxy_scene.update_proxy_view_states(camera, all_fov_proxy_mask[i],
                                             signed_distances=all_sgn_dists[i],
                                             distance_to_surface=None,
                                             X_cam=all_X_cam[i])  # distance_to_surface TO CHANGE!

        # Update the supervision occupancy for proxy points using the signed distance
        proxy_scene.update_proxy_supervision_occ(all_fov_proxy_mask[i], all_sgn_dists[i], tol=params.carving_tolerance)

    # Update the out-of-field status for proxy points inside camera field of view
    proxy_scene.update_proxy_out_of_field(general_fov_proxy_mask)

    # ==================================================================================================================
    # Compute Loss
    # ==================================================================================================================
    # predicted_occs = proxy_scene.proxy_proba[prediction_mask]

    # Occupancy Loss
    if (len(predicted_occs)) > 0:
        occ_loss_computed = True
        if supervise_with_online_field:
            supervision_occs = proxy_scene.proxy_supervision_occ[prediction_mask]
        else:
            supervision_occs = pseudo_gt_proxy_proba[prediction_mask]
        occ_loss = occ_loss_fn(predicted_occs, supervision_occs) \
                   * predicted_occs.shape[0] / params.n_proxy_point_for_occupancy_supervision
    else:
        occ_loss_computed = False
        occ_loss = torch.zeros(1, device=device)[0]

    # Coverage Loss
    if (len(predicted_coverage_gains) > 0):
        cov_loss_computed = True
        cov_loss = cov_loss_fn(predicted_coverage_gains.view(1, -1, 1), supervision_coverage_gains.view(1, -1, 1))
    else:
        cov_loss_computed = False
        cov_loss = torch.zeros(1, device=device)[0]

    # Total loss
    scone_loss = occ_loss + cov_loss

    if print_result and is_master:  # and is_master:
        with torch.no_grad():
            print("----------Memory Replay----------")
            print("--->Occupancy loss:", occ_loss)
            print("--->Coverage loss:", cov_loss)
            print(">Predicted occs sum:", predicted_occs.detach().abs().sum())
            print(">Supervision occs sum:", supervision_occs.detach().sum())
            print(">Predicted cov:", predicted_coverage_gains.detach())
            print(">Supervision cov:", supervision_coverage_gains.detach())
            print(">Visibility gains min/max:", visibility_gains.detach().min(), visibility_gains.detach().max())
            print(">Depth list:", depth_list)

    if occ_loss_computed:
        occ_loss = occ_loss.detach()
    if cov_loss_computed:
        cov_loss = cov_loss.detach()

    new_prediction_camera = all_fov_camera[-1]

    return scone_loss, occ_loss, cov_loss, new_prediction_camera


def recompute_mapping(params, macarons,
                      camera, proxy_scene, surface_scene,
                      device, is_master,
                      save_depths=False,
                      save_depth_every_n_frame=1,  # Not used if save_depth is False
                      depths_memory_path=None,  # Not used if save_depth is False,
                      compute_coarse_mapping=False,
                      return_colors=False
                      ):
    """
    Recomputes the whole mapping of the scene: the depth maps, the backprojected surface partial point cloud, as well as
    the pseudo-GT occupancy values and the view state vectors of the proxy points.

    :param params: (Params)
    :param macarons: (MacaronsWrapper)
    :param camera: (Camera)
    :param proxy_scene: (Scene)
    :param surface_scene: (Scene)
    :param device: (Device)
    :param is_master: (bool)
    :param save_depths: (bool) If True, some of the predicted depth maps are saved in the memory.
    :param save_depth_every_n_frame: (int) Number of frames processed between each saved depth map.
        Unused if save_depth is False.
    :param depths_memory_path: (string) Path to the folder where the depth maps will be saved.
    :param compute_coarse_mapping: (bool) If True, also returns a coarse, simpler, global mapping of the scene.
        Such mapping can be used for warmup supervision of scone modules, at the beginning of training.
    :return: (Tensor)
    """
    if is_master:
        print("\nRecompute mapping...")
        t0 = time.time()
    with torch.no_grad():
        # surface_scene.empty_cells()  # To remove?
        proxy_scene.empty_cells()
        proxy_scene.initialize_proxy_points()

        full_pc = torch.zeros(0, 3, device=device)
        if return_colors:
            full_pc_colors = torch.zeros(0, 3, device=device)
        else:
            full_pc_colors = None

        depth_counter = 0

        if compute_coarse_mapping:
            coarse_surface = surface_scene.return_entire_pt_cloud(return_features=False)
            coarse_surface = coarse_surface[torch.randperm(len(coarse_surface))[:params.coarse_surface_max_size]]
            coarse_mapping_dict = {}
            coarse_mapping_dict['coarse_surface'] = coarse_surface
            coarse_mapping_dict['coarse_proxy_points'] = None
            coarse_mapping_dict['coarse_proxy_probas'] = None
            coarse_mapping_dict['coverage'] = torch.zeros(0, len(coarse_surface))
            coarse_mapping_dict['all_partial_pc'] = []
            coarse_mapping_dict['all_R_cam'] = []
            coarse_mapping_dict['all_T_cam'] = []

        # First, we compute the full surface point cloud from the depth map, and we update the proxy scene.
        for frame_nb in range(params.n_alpha+1, camera.n_frames_captured-1):
            # Load input RGB image and camera pose
            if not params.use_perfect_depth:
                all_images, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(camera=camera,
                                                                                       n_frames=1,
                                                                                       n_alpha=params.n_alpha_for_supervision,
                                                                                       frame_nb=frame_nb)
                all_zbuf = None
            else:
                all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(camera=camera,
                                                                                           n_frames=1,
                                                                                           n_alpha=params.n_alpha_for_supervision,
                                                                                           frame_nb=frame_nb,
                                                                                           return_gt_zbuf=True)

            # Format input as batches to feed depth model
            batch_dict, alpha_dict = create_batch_for_depth_model(params=params,
                                                                  all_images=all_images, all_mask=all_mask,
                                                                  all_R=all_R, all_T=all_T, all_zfar=all_zfar,
                                                                  mode='supervision', device=device, all_zbuf=all_zbuf)

            # Depth prediction
            depth, mask, error_mask, pose, gt_pose = apply_depth_model(params=params,
                                                                       macarons=macarons.depth,
                                                                       batch_dict=batch_dict,
                                                                       alpha_dict=alpha_dict,
                                                                       device=device,
                                                                       use_perfect_depth=params.use_perfect_depth)
            R_cam = batch_dict['R'][0:0 + 1]
            T_cam = batch_dict['T'][0:0 + 1]

            # We fill the surface scene with the partial point cloud
            if depth.shape[0] > 1:
                raise NameError("Problem in remapping.")
            for i in range(depth.shape[0]):
                fov_camera = camera.get_fov_camera_from_RT(R_cam=R_cam,
                                                           T_cam=T_cam)
                # TO CHANGE: filter points based on SSIM value!
                if return_colors:
                    part_pc, part_pc_colors = camera.compute_partial_point_cloud(depth=depth[i:i + 1],
                                                                                images=batch_dict["images"][i:i + 1],
                                                                                mask=(mask * error_mask)[i:i + 1],
                                                                                fov_cameras=fov_camera,
                                                                                gathering_factor=params.gathering_factor,
                                                                                fov_range=params.sensor_range)
                else:
                    part_pc = camera.compute_partial_point_cloud(depth=depth[i:i + 1],
                                                                 mask=(mask * error_mask)[i:i + 1],
                                                                 fov_cameras=fov_camera,
                                                                 gathering_factor=params.gathering_factor,
                                                                 fov_range=params.sensor_range)

                # Fill surface scene
                # part_pc_features = torch.ones(len(part_pc), 1, device=device) # To remove?
                # surface_scene.fill_cells(part_pc, features=part_pc_features) # To remove?

                # Add partial pc to full pc
                full_pc = torch.vstack((full_pc, part_pc))
                if return_colors:
                    full_pc_colors = torch.vstack((full_pc_colors, part_pc_colors))

            # Get Proxy Points in FoV
            fov_proxy_points, fov_proxy_mask = camera.get_points_in_fov(proxy_scene.proxy_points, return_mask=True,
                                                                        fov_camera=fov_camera,
                                                                        fov_range=params.sensor_range)
            # Fill cells with proxy points
            fov_proxy_indices = proxy_scene.get_proxy_indices_from_mask(fov_proxy_mask)
            proxy_scene.fill_cells(fov_proxy_points, features=fov_proxy_indices.view(-1, 1))

            # Computing signed distance of proxy points in fov
            sgn_dists = camera.get_signed_distance_to_depth_maps(pts=fov_proxy_points,
                                                                 depth_maps=depth,
                                                                 mask=mask,
                                                                 fov_camera=fov_camera)
            # Updating view_state vectors
            proxy_scene.update_proxy_view_states(camera, fov_proxy_mask,
                                                 signed_distances=sgn_dists,
                                                 distance_to_surface=None, X_cam=fov_camera.get_camera_center())

            # Update supervision occupancy probabilities
            proxy_scene.update_proxy_supervision_occ(fov_proxy_mask, sgn_dists, tol=params.carving_tolerance)

            # Update the out-of-field status for proxy points inside camera field of view
            proxy_scene.update_proxy_out_of_field(fov_proxy_mask)

            if frame_nb % save_depth_every_n_frame == 0:
                if save_depths:
                    if depths_memory_path is None:
                        raise NameError("Please provide a valid depths_memory_path.")
                    depth_dict = {'depth': depth,
                                  'mask': mask,
                                  'error_mask': error_mask,
                                  'R': R_cam,
                                  'T': T_cam}
                    predicted_depth_save_path = os.path.join(depths_memory_path, str(depth_counter) + '.pt')
                    torch.save(depth_dict, predicted_depth_save_path)
                    depth_counter += 1

                if compute_coarse_mapping:
                    # Compute partial point cloud for current camera
                    max_scene_distance = torch.linalg.norm(surface_scene.x_max - surface_scene.x_min)
                    mask[depth[..., 0] > max_scene_distance] = False
                    partial_pc_j = camera.compute_partial_point_cloud(depth=depth[0:0 + 1],
                                                                      mask=(mask * error_mask)[0:0 + 1],
                                                                      fov_cameras=fov_camera,
                                                                      gathering_factor=2 * params.gathering_factor,
                                                                      fov_range=params.sensor_range)
                    partial_pc_j = surface_scene.get_pts_in_bounding_box(partial_pc_j, return_mask=False)

                    # Coverage computation
                    surface_epsilon = 2. * surface_scene.cell_resolution
                    coverage_j = torch.min(torch.cdist(coarse_surface.double(),
                                                       partial_pc_j.double(), p=2.0), dim=-1)[0]
                    coverage_j = torch.heaviside(surface_epsilon - coverage_j,
                                                 values=torch.zeros_like(coverage_j, device=device))

                    # Final downsampling of partial point cloud
                    partial_pc_j = partial_pc_j[torch.randperm(len(partial_pc_j))[:params.seq_len]]

                    # Fill dictionary
                    coarse_mapping_dict['coverage'] = torch.vstack((coarse_mapping_dict['coverage'],
                                                                    coverage_j))
                    coarse_mapping_dict['all_partial_pc'].append(partial_pc_j)
                    coarse_mapping_dict['all_R_cam'].append(R_cam)
                    coarse_mapping_dict['all_T_cam'].append(T_cam)

        # Then, we fix the surface point cloud and fill the surface scene with it.
        fill_surface_scene(surface_scene, full_pc,
                           random_sampling_max_size=params.n_gt_surface_points,
                           min_n_points_per_cell_fill=3,
                           progressive_fill=params.progressive_fill,
                           max_n_points_per_fill=params.max_points_per_progressive_fill,
                           full_pc_colors=full_pc_colors)

        # Finally, we compute the coarse occupancy field if needed
        if compute_coarse_mapping:
            occupied_mask = proxy_scene.proxy_supervision_occ[..., 0] > 0.
            occupied_proxy_points = proxy_scene.proxy_points[occupied_mask]
            empty_proxy_points = proxy_scene.proxy_points[~occupied_mask]

            occ_field_min_len = min(len(occupied_proxy_points), len(empty_proxy_points))
            coarse_proxy_points = torch.vstack(
                (occupied_proxy_points[torch.randperm(len(occupied_proxy_points))[:occ_field_min_len]],
                 empty_proxy_points[torch.randperm(len(empty_proxy_points))[:occ_field_min_len]]))
            coarse_proxy_probas = torch.zeros(2 * occ_field_min_len, 1, device=device)
            coarse_proxy_probas[:occ_field_min_len] = 1.

            coarse_mapping_dict['coarse_proxy_points'] = coarse_proxy_points
            coarse_mapping_dict['coarse_proxy_probas'] = coarse_proxy_probas

    if is_master:
        tf = time.time()
        print("Total time for remapping:", tf - t0)
    print("Size of pc in surface scene:", surface_scene.return_entire_pt_cloud(return_features=False).shape)

    if compute_coarse_mapping:
        return full_pc, coarse_mapping_dict
    elif return_colors:
        return full_pc, full_pc_colors
    else:
        return full_pc


def save_pose_data(camera, pose_file_path, is_mirrored, scene_scale_factor, mirrored_axis=None):
    dict_to_save = {}
    dict_to_save['scene_scale_factor'] = scene_scale_factor
    dict_to_save['is_mirrored'] = is_mirrored
    dict_to_save['X_cam_history'] = camera.X_cam_history
    dict_to_save['V_cam_history'] = camera.V_cam_history
    if is_mirrored:
        if mirrored_axis is None:
            raise NameError("Please provide the list of mirrored axis.")
        else:
            for axis in mirrored_axis:
                dict_to_save['X_cam_history'][..., axis] = -dict_to_save['X_cam_history'][..., axis]
    torch.save(dict_to_save, pose_file_path)


def train(params,
          dataloader,
          macarons, memory,
          pose_loss_fn, depth_loss_fn, regularity_loss_fn, ssim_loss_fn, occ_loss_fn, cov_loss_fn,
          freeze,
          optimizer, epoch,
          device, is_master,
          train_losses, depth_losses, occ_losses, cov_losses,
          train_coverages):

    num_batches = len(dataloader)
    size = num_batches * params.total_batch_size
    train_loss = torch.zeros(1, device=device)[0]
    avg_depth_loss = torch.zeros(1, device=device)[0]
    avg_occ_loss = torch.zeros(1, device=device)[0]
    avg_cov_loss = torch.zeros(1, device=device)[0]
    train_coverage = torch.zeros(1, device=device)[0]

    # Preparing model
    if params.online_learning:
        macarons.train()
    else:
        macarons.eval()

    warmup_phase = epoch < params.warmup_phase

    if is_master:
        print("Warmup phase:", warmup_phase)

    for batch, scene_dict in enumerate(dataloader):

        scene_names = scene_dict['scene_name']
        obj_names = scene_dict['obj_name']
        all_settings = scene_dict['settings']
        occupied_pose_datas = scene_dict['occupied_pose']

        batch_size = len(scene_names)

        for i_scene in range(batch_size):

            if is_master:
                t0 = time.time()

            scene_name = scene_names[i_scene]
            obj_name = obj_names[i_scene]
            settings = all_settings[i_scene]
            settings = Settings(settings, device, params.scene_scale_factor)
            occupied_pose_data = occupied_pose_datas[i_scene]
            print("\nScene name:", scene_name)
            print("-------------------------------------")

            scene_path = os.path.join(dataloader.dataset.data_path, scene_name)
            mesh_path = os.path.join(scene_path, obj_name)

            mirrored_scene = False
            mirrored_axis = []
            for i in range(len(params.axis_to_mirror)):
                coin_flip = np.random.rand()
                if coin_flip < params.symmetry_probability:
                    mirrored_axis.append(params.axis_to_mirror[i])
                    mirrored_scene = True

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
            depths_path = memory.get_trajectory_depths_path(scene_memory_path, trajectory_nb)
            training_poses_path = memory.get_poses_path(scene_memory_path)
            surface_dir_path = memory.get_trajectory_surface_path(scene_memory_path, trajectory_nb)
            occupancy_dir_path = memory.get_trajectory_occupancy_path(scene_memory_path, trajectory_nb)

            # Setup the Scene and Camera objects
            gt_scene, surface_scene, proxy_scene = setup_scene(params, mesh, settings,
                                                               mirrored_scene, device, is_master,
                                                               mirrored_axis=mirrored_axis)

            camera = setup_camera(params, mesh, proxy_scene, settings, occupied_pose_data,
                                  device, is_master, training_frames_path, mirrored_scene, mirrored_axis=mirrored_axis)

            # Compute curriculum sampling distances for occupancy supervision
            curriculum_distances = get_curriculum_sampling_distances(params, surface_scene, proxy_scene)
            curriculum_n_cells = get_curriculum_sampling_cell_number(params)

            full_pc = [torch.zeros(0, 3, device=device)]

            # Time information
            if params.compute_time:
                loop_time = []
                backward_time = []
                memory_loop_time = []
            else:
                loop_time = None

            if is_master:
                print("Time to setup the scene:", time.time() - t0)

            # Main training loop
            if is_master:
                t0 = time.time()
            for pose_i in range(params.n_poses_in_trajectory):
                # print("Beginning loop", str(pose_i) + "...")

                # ----------Recompute surface---------------------------------------------------------------------------
                if pose_i > 0 and pose_i % params.recompute_surface_every_n_loop == 0:
                    fill_surface_scene(surface_scene, full_pc[0],
                                       random_sampling_max_size=params.n_gt_surface_points,
                                       min_n_points_per_cell_fill=3,
                                       progressive_fill=params.progressive_fill,
                                       max_n_points_per_fill=params.max_points_per_progressive_fill)

                # ----------Main loop-----------------------------------------------------------------------------------
                loss, depth_loss, occ_loss, cov_loss = loop(params, pose_i, mesh, camera,
                                                            surface_scene, proxy_scene,
                                                            curriculum_distances[pose_i], curriculum_n_cells[pose_i],
                                                            macarons, freeze,
                                                            depth_loss_fn, ssim_loss_fn, regularity_loss_fn,
                                                            pose_loss_fn, occ_loss_fn, cov_loss_fn,
                                                            device, is_master,
                                                            full_pc,
                                                            loop_time=loop_time, warmup_phase=warmup_phase)
                # print("Loop is over. Starting backpropagation...")

                # Backpropagation
                if params.use_perfect_depth:
                    optimizer.freeze_depth = True

                if params.compute_time:
                    backward_start_time = time.time()

                if params.online_learning:
                    optimizer.zero_grad()
                    # print("zero grad done.")
                    if not params.use_perfect_depth:
                        depth_loss.backward()

                    if not freeze:
                        loss.backward()
                    # print("backward done.")

                    if pose_i % params.empty_cache_every_n_batch == 0 and is_master and params.check_gradients:
                        if params.jz or params.ddp:
                            check_gradients(macarons.scone.module.occupancy)
                        else:
                            check_gradients(macarons.scone.occupancy)

                    optimizer.step()
                    # print("Optimizer step done.")

                if params.compute_time:
                    backward_end_time = time.time()
                    backward_time.append(backward_end_time - backward_start_time)

                train_loss += loss.detach()
                if params.multiply_loss:
                    train_loss /= params.loss_multiplication_factor
                avg_depth_loss += depth_loss.detach()
                avg_occ_loss += occ_loss
                avg_cov_loss += cov_loss

                # ----------Memory loops--------------------------------------------------------------------------------
                if params.compute_time:
                    memory_loop_start_time = time.time()

                # Memory loop for depth module
                if params.online_learning and memory.current_epoch > 0 and params.n_memory_loops > 0:
                    optimizer.freeze_scone = True
                    for i_loop in range(params.n_memory_loops):
                        depth_loss = memory_loop(params, pose_i, camera, memory,
                                                 macarons, depth_loss_fn, ssim_loss_fn, regularity_loss_fn, pose_loss_fn,
                                                 device, is_master)

                        # print("Memory loop is over. Starting backpropagation...")
                        # Backpropagation
                        optimizer.zero_grad()
                        # print("zero grad done.")
                        depth_loss.backward()
                        # print("backward done.")
                        optimizer.step()
                        # print("Optimizer step done.")

                        avg_depth_loss += depth_loss.detach()
                optimizer.freeze_scone = freeze

                # Memory loop for scone modules
                if params.online_learning and (not freeze) \
                        and (memory.current_epoch > 0) and (params.n_memory_scene_loops > 0):
                    optimizer.freeze_depth = True

                    # Load scene from memory
                    memory_scene_dict = memory.get_random_scene_for_scone_model(params, camera, device,
                                                                                params.n_memory_scene_loops)
                    depths_memory_path = memory_scene_dict['depths_dir_path']
                    memory_surface_scene = memory_scene_dict['partial_surface_scene']
                    memory_total_surface_scene = memory_scene_dict['total_surface_scene']
                    memory_proxy_scene = memory_scene_dict['proxy_scene']
                    memory_pseudo_gt_proxy_proba = memory_scene_dict['pseudo_gt_proxy_proba']
                    memory_prediction_camera = memory_scene_dict['prediction_camera']
                    memory_traj_depth_nb = memory_scene_dict['traj_depth_nb']
                    memory_start_depth_i = memory_scene_dict['start_depth_i']
                    memory_depth_list = memory_scene_dict['depth_list']

                    memory_curriculum_distances = get_curriculum_sampling_distances(params, memory_surface_scene,
                                                                                    memory_proxy_scene)
                    memory_curriculum_n_cells = get_curriculum_sampling_cell_number(params)

                    for i_loop in range(params.n_memory_scene_loops):
                        memory_current_depth_i = memory_start_depth_i + memory_traj_depth_nb - 1 \
                                                 + i_loop * (params.n_interpolation_steps - 1)  # The -1 is crucial!
                        if params.memory_max_curriculum_index == -1:
                            memory_curriculum_index = np.random.randint(len(memory_curriculum_distances))
                        else:
                            memory_curriculum_index = np.random.randint(params.memory_max_curriculum_index)
                        scone_loss, occ_loss, cov_loss, new_prediction_camera = memory_scene_loop(params,
                                                                                                  batch, memory, camera,
                                                                                                  depths_memory_path,
                                                                                                  memory_surface_scene,
                                                                                                  memory_total_surface_scene,
                                                                                                  memory_proxy_scene,
                                                                                                  memory_pseudo_gt_proxy_proba,
                                                                                                  memory_prediction_camera,
                                                                                                  memory_current_depth_i,
                                                                                                  memory_curriculum_distances[memory_curriculum_index],
                                                                                                  memory_curriculum_n_cells[memory_curriculum_index],
                                                                                                  macarons, occ_loss_fn,
                                                                                                  cov_loss_fn,
                                                                                                  device, is_master,
                                                                                                  print_result=(i_loop==params.n_memory_scene_loops-1) and (pose_i % params.empty_cache_every_n_batch==0),
                                                                                                  supervise_with_online_field=params.memory_supervise_with_online_field,
                                                                                                  warmup_phase=warmup_phase,
                                                                                                  depth_list=memory_depth_list)
                        # Backpropagation
                        optimizer.zero_grad()
                        # print("zero grad done.")
                        scone_loss.backward()

                        if pose_i % params.empty_cache_every_n_batch == 0 and is_master and params.check_gradients:
                            if params.jz or params.ddp:
                                check_gradients(macarons.scone.module.occupancy)
                            else:
                                check_gradients(macarons.scone.occupancy)

                        # print("backward done.")
                        optimizer.step()
                        # print("Optimizer step done.")

                        memory_prediction_camera = new_prediction_camera

                        avg_occ_loss += occ_loss
                        avg_cov_loss += cov_loss

                    optimizer.freeze_depth = False
                if params.compute_time:
                    memory_loop_end_time = time.time()
                    memory_loop_time.append(memory_loop_end_time - memory_loop_start_time)

                # Remapping if needed
                if pose_i > 0 and pose_i % params.remap_every_n_poses == 0:
                    full_pc[0] = recompute_mapping(params, macarons,
                                                   camera, proxy_scene, surface_scene,
                                                   device, is_master,
                                                   save_depths=params.n_memory_scene_loops > 0,
                                                   save_depth_every_n_frame=params.save_depth_every_n_frame,
                                                   depths_memory_path=depths_path)
                    if is_master:
                        print("Recomputed full surface point cloud has", len(full_pc[0]), "points.\n")

                # Display info
                if pose_i % params.empty_cache_every_n_batch == 0:

                    # loss = reduce_tensor(loss)
                    if params.ddp or params.jz:
                        # print("Reducing loss...")
                        loss = reduce_tensor(loss+depth_loss, world_size=params.WORLD_SIZE)
                    loss = to_python_float(loss)
                    # print("loss reduced.")

                    current = pose_i  # * idr_torch.size
                    if params.ddp or params.jz:
                        current *= params.WORLD_SIZE

                    if is_master:
                        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]",
                              "computed in", (time.time() - t0) / 60., "minutes.\n")
                        # print(">>>Prediction shape:", pred_depth1.shape,
                        #       "\n>>>Truth norm:", truth_norm, ">>>Prediction norm:", pred_norm, '\n')
                        t0 = time.time()
                    # print("Harmonics:\n", info_harmonics_i)
                        if params.compute_time:
                            print("Avg loop time:", np.mean(loop_time))
                            print("Avg backward time:", np.mean(backward_time))
                            print("Avg memory_loop time:", np.mean(memory_loop_time))

                    # TO REMOVE
                    torch.cuda.empty_cache()


            # Save loss
            if params.ddp or params.jz:
                train_loss = reduce_tensor(train_loss, world_size=params.WORLD_SIZE)
                avg_depth_loss = reduce_tensor(avg_depth_loss, world_size=params.WORLD_SIZE)
                avg_occ_loss = reduce_tensor(avg_occ_loss, world_size=params.WORLD_SIZE)
                avg_cov_loss = reduce_tensor(avg_cov_loss, world_size=params.WORLD_SIZE)

            train_loss = to_python_float(train_loss) / params.n_poses_in_trajectory
            if epoch == 0:
                avg_depth_loss = to_python_float(avg_depth_loss) / params.n_poses_in_trajectory
                avg_occ_loss = to_python_float(avg_occ_loss) / params.n_poses_in_trajectory
                avg_cov_loss = to_python_float(avg_cov_loss) / params.n_poses_in_trajectory
            else:
                avg_depth_loss = to_python_float(avg_depth_loss) / (params.n_poses_in_trajectory
                                                                    * (1 + params.n_memory_loops))
                avg_occ_loss = to_python_float(avg_occ_loss) / (params.n_poses_in_trajectory
                                                                * (1 + params.n_memory_scene_loops))
                avg_cov_loss = to_python_float(avg_cov_loss) / (params.n_poses_in_trajectory
                                                                * (1 + params.n_memory_scene_loops))


            train_losses.append(train_loss)
            depth_losses.append(avg_depth_loss)
            occ_losses.append(avg_occ_loss)
            cov_losses.append(avg_cov_loss)

            train_loss = 0.
            avg_depth_loss = torch.zeros(1, device=device)[0]
            avg_occ_loss = torch.zeros(1, device=device)[0]
            avg_cov_loss = torch.zeros(1, device=device)[0]

            # Save coverage
            train_coverage += gt_scene.scene_coverage(surface_scene)[0]
            if params.ddp or params.jz:
                train_coverage = reduce_tensor(train_coverage, world_size=params.WORLD_SIZE)
            train_coverage = to_python_float(train_coverage)
            train_coverage /= params.n_poses_in_trajectory
            train_coverages.append(train_coverage)

            train_coverage = torch.zeros(1, device=device)[0]

            pose_file_path = os.path.join(training_poses_path, str(epoch) + '.pt')
            save_pose_data(camera, pose_file_path,
                           is_mirrored=mirrored_scene, scene_scale_factor=params.scene_scale_factor,
                           mirrored_axis=mirrored_axis)

            # Save pseudo-GT in memory
            if params.n_memory_scene_loops > 0:
                save_surface_scene_in_memory(surface_dir_path, surface_scene)
                save_occupancy_field_in_memory(occupancy_dir_path, proxy_scene)


def run_training(ddp_rank=None, params=None):
    # Set device
    device = setup_device(params, ddp_rank)

    batch_size = params.batch_size
    total_batch_size = params.total_batch_size

    if params.ddp:
        world_size = params.WORLD_SIZE
        rank = ddp_rank
        is_master = rank == 0
    elif params.jz:
        world_size = idr_torch_size
        rank = idr_torch_rank
        is_master = rank == 0
    else:
        world_size, rank = None, None
        is_master = True

    if params.data_path is None:
        dataset_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../data/scenes")
    else:
        dataset_path = params.data_path

    # Create dataloader
    train_dataloader, _, _ = get_dataloader(train_scenes=params.train_scenes,
                                            val_scenes=params.val_scenes,
                                            test_scenes=params.test_scenes,
                                            batch_size=1,
                                            ddp=params.ddp, jz=params.jz,
                                            world_size=world_size, ddp_rank=rank,
                                            data_path=dataset_path)
    for batch, elem in enumerate(train_dataloader):
        print(elem['scene_name'])

    # Create model
    macarons = load_pretrained_macarons(pretrained_model_path=params.pretrained_model_path,
                                        device=device, learn_pose=params.learn_pose)
    print("The model has", (count_parameters(macarons.depth) + count_parameters(macarons.scone)) / 1e6,
          "trainable parameters.")

    begin_frozen = (params.n_freeze_epochs > 0) and params.start_from_scratch
    if begin_frozen:
        print("Start training with frozen weights for occupancy and visibility modules.")
    else:
        print("Start training with unfrozen weights.")

    if params.start_from_scratch:
        macarons, optimizer, opt_name, start_epoch, best_train_loss = initialize_macarons(params, macarons, device,
                                                                                torch_seed=params.torch_seed,
                                                                                initialize=params.start_from_scratch,
                                                                                pretrained=params.pretrained,
                                                                                ddp_rank=rank,
                                                                                find_unused_parameters=False,
                                                                                load_from_ddp_model=False)
        train_losses = []
        train_coverages = []
        depth_losses = []
        occ_losses = []
        cov_losses = []

    else:
        macarons, optimizer, opt_name, start_epoch, best_train_loss, training_data_dict = initialize_macarons(params,
                                                                                  macarons,
                                                                                  device,
                                                                                  torch_seed=params.torch_seed,
                                                                                  initialize=params.start_from_scratch,
                                                                                  pretrained=params.pretrained,
                                                                                  ddp_rank=rank,
                                                                                  return_training_data=True,
                                                                                  find_unused_parameters=False)
        train_losses = training_data_dict['train_losses']
        train_coverages = training_data_dict['train_coverages']
        depth_losses = training_data_dict['depth_losses']
        # depth_losses = depth_losses[:int(2 * (len(depth_losses) // 2))]
        occ_losses = training_data_dict['occ_losses']
        cov_losses = training_data_dict['cov_losses']
        print("Model reloaded on this GPU with last depth loss", depth_losses[-1])
        print("Length of losses array:", len(train_losses))

    # best_train_loss = 1000
    epochs_without_improvement = 0
    depth_learning_rate = params.depth_learning_rate
    scone_learning_rate = params.scone_learning_rate

    # Readjust learning rate if resume training from checkpoint
    if params.schedule_learning_rate:
        for schedule_epoch in params.depth_lr_epochs:
            if start_epoch > schedule_epoch:
                depth_learning_rate *= params.lr_factor
        print("Initial depth learning rate modified:", depth_learning_rate)
        for schedule_epoch in params.scone_lr_epochs:
            if start_epoch > schedule_epoch:
                scone_learning_rate *= params.lr_factor
        print("Initial scone learning rate modified:", scone_learning_rate)

    # Set loss functions
    pose_loss_fn = get_pose_loss_fn(params)
    regularity_loss_fn = get_regularity_loss_fn(params)
    ssim_loss_fn = None
    if params.training_mode == 'self_supervised':
        depth_loss_fn = get_reconstruction_loss_fn(params)
        ssim_loss_fn = get_ssim_loss_fn(params)
        if is_master:
            print("Model will be trained with self-supervision.")
            print("Value for SSIM loss is set to", params.ssim_factor)
    else:
        raise NameError("Invalid training mode.")

    occ_loss_fn = get_occ_loss_fn(params)
    cov_loss_fn = get_cov_loss_fn(params)

    # Creating memory
    scene_memory_paths = []
    for scene_name in params.train_scenes:
        scene_path = os.path.join(train_dataloader.dataset.data_path, scene_name)
        scene_memory_path = os.path.join(scene_path, params.memory_dir_name)
        scene_memory_paths.append(scene_memory_path)
    memory = Memory(scene_memory_paths=scene_memory_paths, n_trajectories=params.n_memory_trajectories,
                    current_epoch=0)

    if is_master:
        print("Model name:", params.macarons_model_name, "\nArchitecture:\n")
        print(macarons)
        print("Model name:", params.macarons_model_name)
        print("Numbers of trainable parameters:", count_parameters(macarons.depth) + count_parameters(macarons.scone))
        print("Using", opt_name, "optimizer.")

        print(params.n_alpha, "additional frames are used for depth prediction.")
        print("Additional frame indices for prediction are the following:", params.alphas[:params.n_alpha])
        print(params.n_alpha_for_supervision, "additional frames are used for self-supervision.")
        print("Additional frame indices for supervision are the following:", params.alphas)

        print("Training data:", len(train_dataloader), "batches.")
        # print("Validation data:", len(val_dataloader), "batches.")
        print("Batch size:", params.total_batch_size)
        print("Batch size per GPU:", params.batch_size)

        print("Index of axis to be randomly mirrored:", params.axis_to_mirror)

    # Begin training process
    if is_master:
        t0 = time.time()
    for t_e in range(params.epochs):
        t = start_epoch + t_e
        if is_master:
            print("\n-------------------------------------------------------------------------------")
            print(f"Epoch {t + 1}\n-------------------------------")
            print("-------------------------------------------------------------------------------\n")
        torch.cuda.empty_cache()

        # Update model if it began frozen and reach the unfreeze threshold epoch
        freeze = t < params.n_freeze_epochs
        if not freeze and begin_frozen:
            if is_master:
                print("\n===============================================================================")
                print("We now unfreeze all weights.")
                print("===============================================================================\n")
            macarons = load_pretrained_macarons(pretrained_model_path=params.pretrained_model_path,
                                                device=device, learn_pose=params.learn_pose)
            macarons, optimizer, _, _, _ = initialize_macarons(params, macarons, device,
                                                               torch_seed=params.torch_seed,
                                                               initialize=False,
                                                               pretrained=params.pretrained,
                                                               ddp_rank=rank,
                                                               find_unused_parameters=False)
            begin_frozen = False
            best_train_loss = 1000.  # We reset the best loss
        optimizer.freeze_scone = freeze
        if is_master:
            print("Optimizer is frozen:", optimizer.freeze_scone)

        # Update learning rate
        if params.schedule_learning_rate:
            if t in params.depth_lr_epochs:
                print("Multiplying depth learning rate by", params.lr_factor)
                depth_learning_rate *= params.lr_factor
            if t in params.scone_lr_epochs:
                print("Multiplying scone learning rate by", params.lr_factor)
                scone_learning_rate *= params.lr_factor

        update_macarons_learning_rate(params, optimizer,
                                      depth_learning_rate=depth_learning_rate,
                                      scone_learning_rate=scone_learning_rate)

        if is_master:
            print("Max depth learning rate set to", depth_learning_rate)
            print("Max scone learning rate set to", scone_learning_rate)
            print("Current depth learning rate set to", optimizer.depth._rate)
            print("Current scone learning rate set to", optimizer.scone._rate)

        train_dataloader.sampler.set_epoch(t)

        # Update Memory
        memory.current_epoch = t
        print("Memory current epoch is set to", t)

        if params.online_learning:
            macarons.train()
        else:
            macarons.eval()

        train(params,
              train_dataloader,
              macarons, memory,
              pose_loss_fn, depth_loss_fn, regularity_loss_fn, ssim_loss_fn, occ_loss_fn, cov_loss_fn,
              freeze,
              optimizer, t,
              device, is_master,
              train_losses, depth_losses, occ_losses, cov_losses,
              train_coverages)

        loss_array = np.array(depth_losses)
        if len(params.train_scenes) > 8:
            depth_loss_per_epoch = (np.array(loss_array)[::3]
                                    + np.array(loss_array)[1::3]
                                    + np.array(loss_array)[2::3]) / 3.
        elif len(params.train_scenes) > 4:
            depth_loss_per_epoch = (np.array(loss_array)[::2] + np.array(loss_array)[1::2]) / 2.
        else:
            depth_loss_per_epoch = np.array(loss_array)
        current_loss = depth_loss_per_epoch[-1]

        if is_master:
            print("Training done for epoch", t + 1, ".")
            model_save_path = "unvalidated_" + params.macarons_model_name + ".pth"
            model_save_path = os.path.join(weights_dir, model_save_path)
            torch.save({
                'epoch': t + 1,
                'model_state_dict': macarons.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
                'best_loss': best_train_loss,
                'train_losses': train_losses,
                'depth_losses': depth_losses,
                'occ_losses': occ_losses,
                'cov_losses': cov_losses,
                'train_coverages': train_coverages
                # 'val_losses': val_losses,
            }, model_save_path)

            if current_loss < best_train_loss:
                best_model_save_path = "best_unval_" + params.macarons_model_name + ".pth"
                best_model_save_path = os.path.join(weights_dir, best_model_save_path)
                best_train_loss = current_loss
                torch.save({
                    'epoch': t + 1,
                    'model_state_dict': macarons.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': current_loss,
                    'best_loss': best_train_loss,
                    'train_losses': train_losses,
                    'depth_losses': depth_losses,
                    'occ_losses': occ_losses,
                    'cov_losses': cov_losses,
                    'train_coverages': train_coverages
                    # 'val_losses': val_losses,
                }, best_model_save_path)
                print("Best model on training set saved with loss " + str(current_loss) + " .\n")

            if t % params.save_model_every_n_epoch == 0:
                epoch_t_model_name = "epoch_" + str(t) + "_" + params.macarons_model_name + ".pth"
                epoch_t_model_name = os.path.join(weights_dir, epoch_t_model_name)
                torch.save({
                    'epoch': t + 1,
                    'model_state_dict': macarons.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': current_loss,
                    'best_loss': best_train_loss,
                    'train_losses': train_losses,
                    'depth_losses': depth_losses,
                    'occ_losses': occ_losses,
                    'cov_losses': cov_losses,
                    'train_coverages': train_coverages
                    # 'val_losses': val_losses,
                }, epoch_t_model_name)
                print("Model at epoch", t, "saved with loss " + str(current_loss) + " .\n")

        # If depth model suffers from a sudden increase in loss, we reload the previous model
        if params.reload_previous_model_if_bad_loss:
            loss_dt = depth_loss_per_epoch[1:] - depth_loss_per_epoch[:-1]
            if t > 5 and len(loss_dt) > 0 and loss_dt[-1] > params.loss_peak_threshold:
                if is_master:
                    print("\n===============================================================================")
                    print("Sudden increase in depth loss. Reloading previous depth model.")
                    print("===============================================================================\n")
                # macarons = load_pretrained_macarons(pretrained_model_path=params.pretrained_model_path,
                #                                     device=device, learn_pose=params.learn_pose)
                macarons.depth = create_macarons_depth(device)

                # Reload model from last checkpoint epoch:
                epoch_to_load = t // params.save_model_every_n_epoch
                if t % params.save_model_every_n_epoch == 0:
                    epoch_to_load -= 1
                epoch_to_load *= params.save_model_every_n_epoch
                checkpoint_name = "epoch_" + str(epoch_to_load) + "_" + params.macarons_model_name + ".pth"

                macarons, optimizer, _, _, _, training_data_dict = initialize_macarons(params, macarons, device,
                                                                                       torch_seed=params.torch_seed,
                                                                                       initialize=False,
                                                                                       pretrained=params.pretrained,
                                                                                       ddp_rank=rank,
                                                                                       find_unused_parameters=False,
                                                                                       return_training_data=True,
                                                                                       checkpoint_name=checkpoint_name,
                                                                                       previous_optimizer=optimizer,
                                                                                       depth_only=True)
                # train_losses = training_data_dict['train_losses']
                # train_coverages = training_data_dict['train_coverages']
                depth_losses = training_data_dict['depth_losses']
                # occ_losses = training_data_dict['occ_losses']
                # cov_losses = training_data_dict['cov_losses']
                if is_master:
                    print("Model from epoch", epoch_to_load,
                          "reloaded on this GPU with last depth loss", depth_losses[-1])

        torch.cuda.empty_cache()

        # Save data about losses
        if is_master:
            losses_data = {}
            losses_data['train_loss'] = train_losses
            # losses_data['val_loss'] = val_losses
            json_name = "losses_data_" + params.macarons_model_name + ".json"
            with open(json_name, 'w') as outfile:
                json.dump(losses_data, outfile)
            print("Saved data about losses in", json_name, ".")

    if is_master:
        print("Done in", (time.time() - t0) / 3600., "hours!")

        # Save data about losses
        losses_data = {}
        losses_data['train_loss'] = train_losses
        # losses_data['val_loss'] = val_losses
        json_name = "losses_data_" + params.macarons_model_name + ".json"
        with open(json_name, 'w') as outfile:
            json.dump(losses_data, outfile)
        print("Saved data about losses in", json_name, ".")

    if params.ddp or params.jz:
        cleanup()
