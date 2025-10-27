import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
import torch

sys.path.append(os.path.abspath('../'))
from macarons.testers.scene import *
from macarons.utility.render_utils import plot_point_cloud, plot_graph


data_path = "../data/scenes/"

scene_folder = data_path
model_folder = "../weights/macarons"
config_folder = "../configs/macarons"
numGPU = 0
selectable_paths = False

n_3D_points = 20000
point_size = 3
n_frames_to_show = 4
frame_shape = (456, 256)

n_iterations_to_compute = 4

test_resolution = 0.05

use_perfect_depth_map = False  # should be False
compute_collision = False

# magma = plt.get_cmap('magma')
magma_colors = plt.get_cmap('magma')(np.linspace(0, 1, 256))[..., :3]


class GlobalParameters:
    def __init__(self):
        # Selection
        self.scene_folder = scene_folder
        self.model_folder = model_folder
        self.config_folder = config_folder

        self.scene_list = os.listdir(self.scene_folder)
        self.model_list = os.listdir(self.model_folder)
        self.config_list = os.listdir(self.config_folder)

        self.scene_path = ""
        self.model_name = ""
        self.model_path = ""
        self.config_path = ""

        self.loading_ok = False

        # Config parameters object
        self.params = None

        # Dataloader object
        self.dataloader = None

        # Scene-specific parameters and settings
        self.scene_dict = None
        self.scene_name = None
        self.settings = None
        self.occupied_pose_data = None

        # Mesh object and data
        self.mesh = None
        self.obj_name = None
        self.mesh_path = None

        # Memory object
        self.memory = None

        # Model object
        self.model = None

        # Scene objects
        self.gt_scene = None
        self.covered_scene = None
        self.surface_scene = None
        self.proxy_scene = None

        self.full_pc = None
        self.full_pc_colors = None
        self.occupancy_X = None
        self.occupancy_sigma = None

        self.curriculum_distances = None
        self.curriculum_n_cells = None

        # Camera object
        self.start_cam_idx = None
        self.camera = None
        self.pose_i = 0

        # Frames
        self.current_frames = None
        self.current_depths = None
        self.current_mask = None

        # Plot state
        self.plot_ok = False

        # Coverage metric
        self.coverage_evolution = np.zeros(0)

        # Device
        self.device = None

        self.computation_done = False
        self.update_board = False


global_params = GlobalParameters()

# ----------Selection tab-----------------------------------------------------------------------------------------------


def refresh_scene_folder(new_folder):
    files_list = os.listdir(new_folder)
    global_params.scene_folder = new_folder
    return gr.Dropdown.update(choices=files_list)


def refresh_model_folder(new_folder):
    files_list = os.listdir(new_folder)
    global_params.model_folder = new_folder
    return gr.Dropdown.update(choices=files_list)


def refresh_config_folder(new_folder):
    files_list = os.listdir(new_folder)
    global_params.config_folder = new_folder
    return gr.Dropdown.update(choices=files_list)


def load_selected(scene_name, model_name, config_name):
    if min(len(scene_name), len(model_name), len(config_name)) == 0:
        return "<p style='color:#dc2626'>" \
               "<b style='color:#dc2626'>Error:</b> Please select a scene, a model and a config file." \
               "</p>"
    else:
        # Update selection parameters
        global_params.scene_path = os.path.join(global_params.scene_folder, scene_name)
        global_params.model_path = os.path.join(global_params.model_folder, model_name)
        global_params.model_name = model_name
        global_params.config_path = os.path.join(global_params.config_folder, config_name)

        # Load parameters from config file
        global_params.params = load_params(global_params.config_path)

        # Setup device
        # device = 'cpu'
        global_params.params.jz = False
        global_params.params.ddp = False
        global_params.params.numGPU = numGPU
        global_params.device = setup_device(global_params.params, None)

        # Load model and create dataloader
        global_params.params.data_path = global_params.scene_folder
        global_params.params.test_scenes = [scene_name]
        test_dataloader, model, memory = setup_test(params=global_params.params,
                                                    model_path=global_params.model_path,
                                                    device=global_params.device)
        global_params.dataloader = test_dataloader
        global_params.model = model
        global_params.memory = memory

        # Load 3D mesh and scene data
        global_params.mesh = None
        torch.cuda.empty_cache()
        global_params.scene_dict = global_params.dataloader.dataset[0]
        global_params.scene_name = global_params.scene_dict['scene_name']
        global_params.obj_name = global_params.scene_dict['obj_name']
        global_params.settings = global_params.scene_dict['settings']
        global_params.settings = Settings(global_params.settings, global_params.device,
                                          global_params.params.scene_scale_factor)
        global_params.occupied_pose_data = global_params.scene_dict['occupied_pose']

        global_params.mesh_path = os.path.join(global_params.scene_path, global_params.obj_name)

        # Load mesh
        global_params.mesh = load_scene(global_params.mesh_path, global_params.params.scene_scale_factor,
                                        global_params.device, mirror=False, mirrored_axis=None)

        # Memory info
        global_params.scene_memory_path = os.path.join(global_params.scene_path,
                                                       global_params.params.memory_dir_name)
        global_params.trajectory_nb = global_params.memory.current_epoch % global_params.memory.n_trajectories
        global_params.training_frames_path = global_params.memory.get_trajectory_frames_path(global_params.scene_memory_path,
                                                                                             global_params.trajectory_nb)
        global_params.training_poses_path = global_params.memory.get_poses_path(global_params.scene_memory_path)

        # Construct scene objects
        gt_scene, covered_scene, surface_scene, proxy_scene = setup_test_scene(global_params.params,
                                                                               global_params.mesh,
                                                                               global_params.settings,
                                                                               mirrored_scene=False,
                                                                               device=global_params.device,
                                                                               mirrored_axis=None,
                                                                               surface_scene_feature_dim=3,
                                                                               test_resolution=test_resolution
                                                                               )
        global_params.gt_scene = gt_scene
        global_params.covered_scene = covered_scene
        global_params.surface_scene = surface_scene
        global_params.proxy_scene = proxy_scene

        # Construct camera
        start_cam_idx_i = np.random.randint(len(global_params.settings.camera.start_positions))
        global_params.start_cam_idx = global_params.settings.camera.start_positions[start_cam_idx_i]
        global_params.camera = setup_test_camera(global_params.params,
                                                 global_params.mesh,
                                                 global_params.start_cam_idx,
                                                 global_params.settings,
                                                 global_params.occupied_pose_data,
                                                 global_params.device,
                                                 global_params.training_frames_path,
                                                 mirrored_scene=False,
                                                 mirrored_axis=None)

        # Initialize trajectory
        global_params.model.eval()
        global_params.curriculum_distances = get_curriculum_sampling_distances(global_params.params,
                                                                               global_params.surface_scene,
                                                                               global_params.proxy_scene)
        global_params.curriculum_n_cells = get_curriculum_sampling_cell_number(global_params.params)

        global_params.full_pc = torch.zeros(0, 3, device=global_params.device)
        global_params.full_pc_colors = torch.zeros(0, 3, device=global_params.device)
        global_params.coverage_evolution = []

        res = "<p>Loading successful."
        res += "<br><br>Loaded model has " + str((count_parameters(global_params.model.depth)
                                                 + count_parameters(global_params.model.scone)) / 1e6) \
               + " M trainable parameters."

        res += "<br><br> Memory path used for demo: " + str(global_params.scene_memory_path)

        # res += "<br><br>Mesh Vertices shape:" + str(global_params.mesh.verts_list()[0].shape)
        # res += "<br>Min Vert:" + str(torch.min(global_params.mesh.verts_list()[0], dim=0)[0])
        # res += "<br>Max Vert:" + str(torch.max(global_params.mesh.verts_list()[0], dim=0)[0])
        #
        # res += "<br><br>Folders:<br>" \
        #        + global_params.scene_folder + "<br>" \
        #        + global_params.model_folder + "<br>" \
        #        + global_params.config_folder
        # res += "<br><br>Names:<br>" + scene_name + '<br>' + model_name + '<br>' + config_name
        res += "<br><br>Please go to the <tt style='color:#ea580c'>Exploration</tt> tab to explore " \
               "and reconstruct the 3D scene using the loaded model."
        res += "</p>"

        global_params.loading_ok = True

        return res


def compute_next_camera_pose():
    global_params.computation_done = False
    global_params.update_board = False

    for i_iteration_to_compute in range(n_iterations_to_compute):
        if global_params.pose_i > 0 and global_params.pose_i % global_params.params.recompute_surface_every_n_loop == 0:
            print("Recomputing surface...")
            fill_surface_scene(global_params.surface_scene, global_params.full_pc,
                               random_sampling_max_size=global_params.params.n_gt_surface_points,
                               min_n_points_per_cell_fill=3,
                               progressive_fill=global_params.params.progressive_fill,
                               max_n_points_per_fill=global_params.params.max_points_per_progressive_fill,
                               full_pc_colors=global_params.full_pc_colors)

        # ----------Predict visible surface points from RGB images------------------------------------------------------

        # Load input RGB image and camera pose
        all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(camera=global_params.camera,
                                                                                             n_frames=1,
                                                                                             n_alpha=global_params.params.n_alpha,
                                                                                             return_gt_zbuf=True)
        # Register GT surface points to compute true coverage for evaluation
        for i in range(all_zbuf[-1:].shape[0]):
            # TO CHANGE: filter points based on SSIM value!
            part_pc = global_params.camera.compute_partial_point_cloud(depth=all_zbuf[-1:],
                                                         mask=all_mask[-1:],
                                                         fov_cameras=global_params.camera.get_fov_camera_from_RT(
                                                             R_cam=all_R[-1:],
                                                             T_cam=all_T[-1:]),
                                                         gathering_factor=global_params.params.gathering_factor,
                                                         fov_range=global_params.params.sensor_range)

            # Fill surface scene
            part_pc_features = torch.zeros(len(part_pc), 1, device=global_params.device)
            global_params.covered_scene.fill_cells(part_pc, features=part_pc_features)  # Todo: change for better surface
        # Compute true coverage for evaluation
        current_coverage = global_params.gt_scene.scene_coverage(global_params.covered_scene,
                                                                 surface_epsilon=2 * test_resolution * global_params.params.scene_scale_factor)

        if current_coverage[0] == 0.:
            global_params.coverage_evolution.append(0.)
        else:
            global_params.coverage_evolution.append(current_coverage[0].item())

        surface_distance = global_params.curriculum_distances[global_params.pose_i]  # todo: what if pose_i >= len(curr...)?

        # Format input as batches to feed depth model
        batch_dict, alpha_dict = create_batch_for_depth_model(params=global_params.params,
                                                              all_images=all_images, all_mask=all_mask,
                                                              all_R=all_R, all_T=all_T, all_zfar=all_zfar,
                                                              mode='inference', device=global_params.device,
                                                              all_zbuf=all_zbuf)

        # Depth prediction
        with torch.no_grad():
            depth, mask, error_mask, pose, gt_pose = apply_depth_model(params=global_params.params,
                                                                       macarons=global_params.model.depth,
                                                                       batch_dict=batch_dict,
                                                                       alpha_dict=alpha_dict,
                                                                       device=global_params.device,
                                                                       use_perfect_depth=global_params.params.use_perfect_depth)

        if use_perfect_depth_map:
            depth = all_zbuf[2:3]
            error_mask = mask

        # We fill the surface scene with the partial point cloud
        for i in range(depth.shape[0]):
            # TO CHANGE: filter points based on SSIM value!
            part_pc, part_pc_features = global_params.camera.compute_partial_point_cloud(depth=depth[i:i + 1],
                                                         images=batch_dict["images"][i:i+1],
                                                         mask=(mask * error_mask)[i:i + 1],
                                                         fov_cameras=global_params.camera.get_fov_camera_from_RT(
                                                             R_cam=batch_dict['R'][i:i + 1],
                                                             T_cam=batch_dict['T'][i:i + 1]),
                                                         gathering_factor=global_params.params.gathering_factor,
                                                         fov_range=global_params.params.sensor_range)

            # Fill surface scene
            # part_pc_features = torch.zeros(len(part_pc), 1, device=global_params.device)
            global_params.surface_scene.fill_cells(part_pc, features=part_pc_features)

            global_params.full_pc = torch.vstack((global_params.full_pc, part_pc))
            global_params.full_pc_colors = torch.vstack((global_params.full_pc_colors, part_pc_features))

        # ----------Update Proxy Points data with current FoV-----------------------------------------------------------

        # Get Proxy Points in current FoV
        fov_proxy_points, fov_proxy_mask = global_params.camera.get_points_in_fov(global_params.proxy_scene.proxy_points,
                                                                                  return_mask=True,
                                                                                  fov_camera=None,
                                                                                  fov_range=global_params.params.sensor_range)
        fov_proxy_indices = global_params.proxy_scene.get_proxy_indices_from_mask(fov_proxy_mask)
        global_params.proxy_scene.fill_cells(fov_proxy_points, features=fov_proxy_indices.view(-1, 1))

        # Computing signed distance of proxy points in fov
        sgn_dists = global_params.camera.get_signed_distance_to_depth_maps(pts=fov_proxy_points,
                                                                           depth_maps=depth,
                                                                           mask=mask, fov_camera=None)

        # Updating view_state vectors
        global_params.proxy_scene.update_proxy_view_states(global_params.camera, fov_proxy_mask,
                                                           signed_distances=sgn_dists,
                                                           distance_to_surface=None, X_cam=None)  # distance_to_surface TO CHANGE!

        # Update the supervision occupancy for proxy points using the signed distance
        global_params.proxy_scene.update_proxy_supervision_occ(fov_proxy_mask, sgn_dists,
                                                               tol=global_params.params.carving_tolerance)

        # Update the out-of-field status for proxy points inside camera field of view
        global_params.proxy_scene.update_proxy_out_of_field(fov_proxy_mask)

        # Update visibility history of surface points
        # global_params.surface_scene.set_all_features_to_value(value=1.)

        # ----------Predict Occupancy Probability Field-----------------------------------------------------------------

        with torch.no_grad():
            X_world, view_harmonics, occ_probs = compute_scene_occupancy_probability_field(global_params.params,
                                                                                           global_params.model.scone,
                                                                                           global_params.camera,
                                                                                           global_params.surface_scene,
                                                                                           global_params.proxy_scene,
                                                                                           global_params.device)

        global_params.occupancy_X = X_world + 0.
        global_params.occupancy_sigma = occ_probs + 0.

        # ----------Predict Coverage Gain of neighbor camera poses----------------------------------------------------------

        # Compute valid neighbor poses
        neighbor_indices = global_params.camera.get_neighboring_poses()
        valid_neighbors = global_params.camera.get_valid_neighbors(neighbor_indices=neighbor_indices,
                                                                   mesh=global_params.mesh)

        max_coverage_gain = -1.
        next_idx = valid_neighbors[0]

        # For each valid neighbor...
        for neighbor_i in range(len(valid_neighbors)):
            neighbor_idx = valid_neighbors[neighbor_i]
            neighbor_pose, _ = global_params.camera.get_pose_from_idx(neighbor_idx)
            X_neighbor, V_neighbor, fov_neighbor = global_params.camera.get_camera_parameters_from_pose(neighbor_pose)

            # We check, if needed, if camera collides
            drop_neighbor = False
            if compute_collision:
                drop_neighbor = global_params.proxy_scene.camera_collides(global_params.params, global_params.camera, X_neighbor)

            if not drop_neighbor:
                # ...We predict its coverage gain...
                with torch.no_grad():
                    _, _, visibility_gains, coverage_gain = predict_coverage_gain_for_single_camera(
                        params=global_params.params, macarons=global_params.model.scone,
                        proxy_scene=global_params.proxy_scene, surface_scene=global_params.surface_scene,
                        X_world=X_world, proxy_view_harmonics=view_harmonics, occ_probs=occ_probs,
                        camera=global_params.camera, X_cam_world=X_neighbor, fov_camera=fov_neighbor)

                # ...And save it with the neighbor index if the estimated coverage is better
                if coverage_gain.shape[0] > 0 and coverage_gain > max_coverage_gain:
                    max_coverage_gain = coverage_gain
                    next_idx = neighbor_idx

        X_cam_t = 0. + global_params.camera.X_cam
        V_cam_t = 0. + global_params.camera.V_cam
        fov_camera_t = global_params.camera.get_fov_camera_from_XV(X_cam=X_cam_t, V_cam=V_cam_t)

        # ==================================================================================================================
        # Move to the neighbor NBV and acquire supervision signal
        # ==================================================================================================================

        # Now that we have estimated the NBV among neighbors, we move toward this new camera pose and save RGB images along
        # the way.

        # ----------Move to next camera pose--------------------------------------------------------------------------------
        # We move to the next pose and capture RGB images.
        interpolation_step = 1
        for i in range(global_params.camera.n_interpolation_steps):
            global_params.camera.update_camera(next_idx, interpolation_step=interpolation_step)
            global_params.camera.capture_image(global_params.mesh)
            interpolation_step += 1

        # ----------Depth prediction------------------------------------------------------------------------------------
        # Load input RGB image and camera pose
        all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(camera=global_params.camera,
                                                                                             n_frames=global_params.params.n_interpolation_steps,
                                                                                             n_alpha=global_params.params.n_alpha_for_supervision,
                                                                                             return_gt_zbuf=True)
        # Format input as batches to feed depth model
        batch_dict, alpha_dict = create_batch_for_depth_model(params=global_params.params,
                                                              all_images=all_images, all_mask=all_mask,
                                                              all_R=all_R, all_T=all_T, all_zfar=all_zfar,
                                                              mode='supervision', device=global_params.device, all_zbuf=all_zbuf)

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
                depth_i, mask_i, error_mask_i, _, _ = apply_depth_model(params=global_params.params, macarons=global_params.model.depth,
                                                                        batch_dict=batch_dict_i,
                                                                        alpha_dict=alpha_dict_i,
                                                                        device=global_params.device,
                                                                        compute_loss=False,
                                                                        use_perfect_depth=global_params.params.use_perfect_depth)
                if use_perfect_depth_map:
                    depth_i = all_zbuf[2 + i:3 + i]
                    error_mask_i = mask_i

            depth.append(depth_i)
            mask.append(mask_i)
            error_mask.append(error_mask_i)
        depth = torch.cat(depth, dim=0)
        mask = torch.cat(mask, dim=0)
        error_mask = torch.cat(error_mask, dim=0)

        # Save current images and depths
        global_params.current_frames = batch_dict['images'] + 0.
        global_params.current_depths = depth + 0.
        global_params.current_mask = mask
        if depth.get_device() > -1:
            global_params.current_frames = global_params.current_frames.cpu()
            global_params.current_depths = global_params.current_depths.cpu()
            global_params.current_mask = global_params.current_mask.cpu()

        # ----------Build supervision signal from the new depth maps----------------------------------------------------
        all_part_pc = []
        all_part_pc_features = []
        all_fov_proxy_points = torch.zeros(0, 3, device=global_params.device)
        general_fov_proxy_mask = torch.zeros(global_params.params.n_proxy_points, device=global_params.device).bool()
        all_fov_proxy_mask = []
        all_sgn_dists = []
        all_X_cam = []
        all_fov_camera = []

        close_fov_proxy_mask = torch.zeros(global_params.params.n_proxy_points, device=global_params.device).bool()

        for i in range(depth.shape[0]):
            fov_frame = global_params.camera.get_fov_camera_from_RT(R_cam=batch_dict['R'][i:i + 1], T_cam=batch_dict['T'][i:i + 1])
            all_X_cam.append(fov_frame.get_camera_center())
            all_fov_camera.append(fov_frame)

            # TO CHANGE: filter points based on SSIM value!
            part_pc, part_pc_features = global_params.camera.compute_partial_point_cloud(depth=depth[i:i + 1],
                                                                       images=batch_dict['images'][i:i+1],
                                                                       mask=(mask * error_mask)[i:i + 1].bool(),
                                                                       fov_cameras=fov_frame,
                                                                       gathering_factor=global_params.params.gathering_factor,
                                                                       fov_range=global_params.params.sensor_range)

            # Surface points to fill surface scene
            all_part_pc.append(part_pc)
            all_part_pc_features.append(part_pc_features)

            # Get Proxy Points in current FoV
            fov_proxy_points, fov_proxy_mask = global_params.camera.get_points_in_fov(global_params.proxy_scene.proxy_points, return_mask=True,
                                                                        fov_camera=fov_frame, fov_range=global_params.params.sensor_range)
            all_fov_proxy_points = torch.vstack((all_fov_proxy_points, fov_proxy_points))
            all_fov_proxy_mask.append(fov_proxy_mask)
            general_fov_proxy_mask = general_fov_proxy_mask + fov_proxy_mask

            # Computing signed distance of proxy points in fov
            sgn_dists = global_params.camera.get_signed_distance_to_depth_maps(pts=fov_proxy_points, depth_maps=depth[i:i + 1],
                                                                 mask=mask[i:i + 1].bool(), fov_camera=fov_frame
                                                                 ).view(-1, 1)
            all_sgn_dists.append(sgn_dists)

            # Computing mask for proxy points close to the surface. We will use this for occupancy probability supervision.
            close_fov_proxy_mask[fov_proxy_mask] = False + (sgn_dists.view(-1).abs() < surface_distance)

        # ----------Update Scenes to finalize supervision signal and prepare next iteration---------------------------------

        # 1. Surface scene
        # Fill surface scene
        # We give a visibility=1 to points that were visible in frame t, and 0 to others
        complete_part_pc = torch.vstack(all_part_pc)
        complete_part_pc_features = torch.vstack(all_part_pc_features) # torch.zeros(len(complete_part_pc), 1, device=global_params.device)
        # complete_part_pc_features[:len(all_part_pc[0])] = 1.
        global_params.surface_scene.fill_cells(complete_part_pc, features=complete_part_pc_features)

        global_params.full_pc = torch.vstack((global_params.full_pc, complete_part_pc))
        global_params.full_pc_colors = torch.vstack((global_params.full_pc_colors, complete_part_pc_features))

        # Compute coverage gain for each new camera pose
        # We also include, at the beginning, the previous camera pose with a coverage gain equal to 0.
        # supervision_coverage_gains = torch.zeros(global_params.params.n_interpolation_steps, 1, device=global_params.device)
        # for i in range(depth.shape[0]):
        #     supervision_coverage_gains[i, 0] = global_params.surface_scene.camera_coverage_gain(all_part_pc[i],
        #                                                                                         surface_epsilon=None)

        # Update visibility history of surface points
        # global_params.surface_scene.set_all_features_to_value(value=1.)

        # 2. Proxy scene
        # Fill proxy scene
        general_fov_proxy_indices = global_params.proxy_scene.get_proxy_indices_from_mask(general_fov_proxy_mask)
        global_params.proxy_scene.fill_cells(global_params.proxy_scene.proxy_points[general_fov_proxy_mask],
                                             features=general_fov_proxy_indices.view(-1, 1))

        for i in range(depth.shape[0]):
            # Updating view_state vectors
            global_params.proxy_scene.update_proxy_view_states(global_params.camera, all_fov_proxy_mask[i],
                                                               signed_distances=all_sgn_dists[i],
                                                               distance_to_surface=None,
                                                               X_cam=all_X_cam[i])  # distance_to_surface TO CHANGE!

            # Update the supervision occupancy for proxy points using the signed distance
            global_params.proxy_scene.update_proxy_supervision_occ(all_fov_proxy_mask[i], all_sgn_dists[i],
                                                                   tol=global_params.params.carving_tolerance)

        # Update the out-of-field status for proxy points inside camera field of view
        global_params.proxy_scene.update_proxy_out_of_field(general_fov_proxy_mask)

        global_params.pose_i += 1

        print("pose_i:", global_params.pose_i)
    global_params.computation_done = True
    return update_explore_message()


def plot_3D_scene(checkboxes):
    # X = torch.rand(20000, 3)
    X = torch.zeros(0, 3)
    c = torch.zeros(0, 3)

    global_params.plot_ok = False

    if "GT surface" in checkboxes:
        X, c = global_params.gt_scene.return_entire_pt_cloud(return_features=True)
        # c = 0.5 * torch.ones_like(X)
        global_params.plot_ok = True

    if "Reconstructed surface" in checkboxes:  # todo: replace with full_pc, and remove points outside bounding box?
        X, c = global_params.surface_scene.return_entire_pt_cloud(return_features=True)
        global_params.plot_ok = True

    if "Occupancy field" in checkboxes:
        occ_mask = global_params.occupancy_sigma[..., 0] > 0.5
        X = global_params.occupancy_X[occ_mask]
        # c = torch.zeros_like(X)
        # c[..., 2] = global_params.occupancy_sigma[..., 0][occ_mask]

        c = global_params.occupancy_sigma[..., 0][occ_mask]

        if len(c) > 0:
            c = (c - c.min()) / (c.max() - c.min() + 1e-7)
            colors = torch.Tensor(plt.get_cmap('magma')(np.linspace(0, 1, 256))[..., :3]).to(global_params.device)
            c = colors[(255 * c.clamp(0., 1.).view(-1)).long()]

            global_params.plot_ok = True
        else:
            global_params.plot_ok = False

    if not global_params.plot_ok:
        X = torch.zeros(1, 3, device=global_params.device)
        c = torch.ones(1, 3, device=global_params.device)

    if "Camera trajectory" in checkboxes:
        X_cam = global_params.camera.X_cam_history + 0.
        c_cam = torch.zeros_like(X_cam)
        c_cam[..., 0] += 1
        c_cam[..., 1] += torch.linspace(0.9, 0.0, len(X_cam), device=global_params.device)
        c_cam[..., 2] += torch.linspace(0.9, 0.0, len(X_cam), device=global_params.device)

        X_perm = torch.randperm(len(X))
        X = torch.cat((X_cam, X[X_perm][:n_3D_points - len(X_cam)]), dim=0)
        c = torch.cat((c_cam, c[X_perm][:n_3D_points - len(c_cam)]), dim=0)
        global_params.plot_ok = True

    return plot_point_cloud(X, c, name="3D Scene", max_points=n_3D_points, point_size=point_size,
                            width=730, height=600)


def plot_coverage():
    x = np.arange(len(global_params.coverage_evolution))
    return plot_graph(x=x, y=np.array(global_params.coverage_evolution),
                      x_label="NBV iterations", y_label="Total surface coverage",
                      width=730, height=250)


def update_frames():
    res = []
    for i in range(min(n_frames_to_show, global_params.params.n_interpolation_steps)):
        res.append(global_params.current_frames[i].numpy())

    return res


def update_depths():
    res = []
    for i in range(min(n_frames_to_show, global_params.params.n_interpolation_steps)):
        res_i = 1. / global_params.current_depths[i]  # todo: clamp between 0 and 1
        res_i[~global_params.current_mask[i]] = 0.
        res_i = (res_i - res_i.min()) / (res_i.max() - res_i.min())
        res_i = res_i.clamp(0., 1.)[..., 0]
        res_i = magma_colors[(res_i.numpy() * 255).astype(np.uint)]
        res.append(res_i)
        # print(res_i, type(res_i), res_i.shape)
        # print(global_params.current_depths.shape)

    return res


def update_explore_message():
    res = ""
    global_params.update_board = False
    if not global_params.loading_ok:
        res += "<b style='color:#dc2626'>Error:</b> Please first select a 3D scene, a model and a config file " \
               "in the <tt style='color:#ea580c'>Selection</tt> tab."

    elif global_params.computation_done:
        res += "Computation done for pose " + str(global_params.pose_i - 1) + ". "
        res += "Total surface coverage has reached " + str(global_params.coverage_evolution[-1])
        global_params.update_board = True
        global_params.computation_done = False
    return res


# ----------Main--------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to launch a gradio demo showing how macarons works.')
    parser.add_argument('-d', '--device', type=int, help='Number of the GPU device to use for the demo.')

    args = parser.parse_args()

    if args.device:
        numGPU = args.device

    print("Using GPU device " + str(numGPU) + '.')

    with gr.Blocks() as demo:
        # Header text
        gr.HTML("<center>"
                "<font size='6'> "
                    "<b>MACARONS: Mapping And Coverage Anticipation with RGB ONline Self-supervision</b>"
                "</font><br>"
                "<font size='5'>"
                    "CVPR 2023"
                "</font><br><br>"
                "<font size='4'> "
                    "<b>Antoine Guédon, Tom Monnier, Pascal Monasse, Vincent Lepetit</b><br>"
                    "LIGM, Ecole des Ponts, Univ Gustave Eiffel, CNRS"
                "</font>"
                )

        gr.HTML("Please select a 3D scene, a model and a config file using the <tt style='color:#ea580c'>Selection</tt> tab.<br>"
                "Then, use the <tt style='color:#ea580c'>Exploration</tt> tab to explore and reconstruct the selected 3D scene.")

        # First tab: Select a 3D scene and model------------------------------------------------------------------------
        with gr.Tab("Selection"):

            # Select a 3D scene
            gr.HTML("<h2>1. Select a 3D scene</h2>")
            if selectable_paths:
                gr.HTML(# "Path to 3D scene folder.<br>"
                        "<p style='color:#9ca3af'>"
                            "Enter the path to the folder containing all 3D scenes subfolders, and press the Enter key."
                        "</p>"
                        #"<small style='color:#9ca3af'>Enter the path to the folder containing all 3D scenes subfolders.</small>"
                        )
                scene_text_input = gr.Textbox(#label="Path to 3D scene folder.",
                                              #info="Enter the path to the folder containing all 3D scenes subfolders.",
                                              value=global_params.scene_folder,
                                              placeholder="scenes...",
                                              show_label=False)
                scene_text_input.style(container=False)

            gr.HTML("<p style='color:#9ca3af'>"
                        "Please use the following dropdown list to select a scene to explore."
                    "</p>")
            scene_dropdown = gr.Dropdown(choices=global_params.scene_list,
                                         #label="Select a scene to explore.",
                                         #info="Select a scene to explore.",
                                         # value=global_params.scene_list[0],
                                         max_choices=1, multiselect=False, interactive=True, show_label=False)
            scene_dropdown.style(container=False)

            # Select a MACARONS model
            gr.HTML(#"<br>"
                    "<h2>2. Select a model</h2>")
            if selectable_paths:
                gr.HTML("<p style='color:#9ca3af'>"
                            "Enter the path to the folder containing all NBV models, and press the Enter key.<br>"
                            "Models should have .pth extension."
                        "</p>")
                model_text_input = gr.Textbox(
                    value=global_params.model_folder,
                    placeholder="models...",
                    show_label=False)
                model_text_input.style(container=False)

            gr.HTML("<p style='color:#9ca3af'>"
                    "Please use the following dropdown list to select a model for exploration."
                    "</p>")
            model_dropdown = gr.Dropdown(choices=global_params.model_list,
                                         # info="Select a model for exploration.",
                                         max_choices=1, multiselect=False, interactive=True, show_label=False)
            model_dropdown.style(container=False)

            # Select a config file
            gr.HTML(#"<br>"
                    "<h2>3. Select a config file</h2>")
            if selectable_paths:
                gr.HTML("<p style='color:#9ca3af'>"
                            "Enter the path to the folder containing all config JSON files, and press the Enter key.<br>"
                            "You should choose a config file that fits the selected model."
                            # "Config files should have .json extension.<br>"
                        "</p>")
                config_text_input = gr.Textbox(
                    value=global_params.config_folder,
                    placeholder="configs...",
                    show_label=False)
                config_text_input.style(container=False)

            gr.HTML("<p style='color:#9ca3af'>"
                    "Please use the following dropdown list to select a config file for exploration."
                    "</p>")
            config_dropdown = gr.Dropdown(choices=global_params.config_list,
                                          max_choices=1, multiselect=False, interactive=True, show_label=False)
            config_dropdown.style(container=False)

            # text_output = gr.Textbox()

            # Buttons
            # refresh_button = gr.Button("Refresh")
            load_button = gr.Button("Load")
            load_text = gr.HTML("")

        # Second tab: Explore a 3D scene--------------------------------------------------------------------------------
        explore_tab = gr.Tab("Exploration")
        with explore_tab:
            with gr.Row():
                #image_input = gr.Image()
                with gr.Column():
                    # gr.HTML("Plot either the GT surface points or the reconstructed surface points.")

                    # Plot options
                    plot_scene_options = gr.CheckboxGroup(choices=["GT surface",
                                                                   "Camera trajectory",
                                                                   "Reconstructed surface",
                                                                   "Occupancy field"
                                                                   ],
                                                          value=["GT surface", "Camera trajectory"],
                                                          info="Elements to display.",
                                                          interactive=True,
                                                          show_label=False
                                                          )
                    plot_scene_options.style()
                    # gr.HTML("Plot the computed trajectory.")

                    # Frame images
                    black_frame = np.zeros((frame_shape[1], frame_shape[0], 3))
                    frame_variant = "panel"  # "panel", "compact"
                    with gr.Row(variant=frame_variant):
                        frame_images = []
                        for i in range(n_frames_to_show):
                            if i < n_frames_to_show-1:
                                frame_label = "Frame at t-" + str(n_frames_to_show-1-i)
                            else:
                                frame_label = "Frame at t"
                            frame_image_i = gr.Image(value=black_frame, label=frame_label)
                            frame_images.append(frame_image_i)

                    # Depth images
                    with gr.Row(variant=frame_variant):
                        depth_images = []
                        for i in range(n_frames_to_show):
                            if i < n_frames_to_show-1:
                                frame_label = "Depth at t-" + str(n_frames_to_show-1-i)
                            else:
                                frame_label = "Depth at t"
                            depth_image_i = gr.Image(value=black_frame, label=frame_label)
                            depth_images.append(depth_image_i)

                    coverage_plot_output = gr.Plot()

                # image_output = gr.Image()
                scene_plot_output = gr.Plot()
            explore_message = gr.HTML("")
            nbv_button = gr.Button("Compute next camera poses")

        # with gr.Accordion("3D scenes"):
        #     gr.Markdown("To do...")

        # ----------Selection tab---------------------------------------------------------------------------------------
        if selectable_paths:
            scene_text_input.submit(refresh_scene_folder,
                                    inputs=scene_text_input,
                                    outputs=scene_dropdown)
            model_text_input.submit(refresh_model_folder,
                                    inputs=model_text_input,
                                    outputs=model_dropdown)
            config_text_input.submit(refresh_config_folder,
                                     inputs=config_text_input,
                                     outputs=config_dropdown)

        load_button.click(load_selected,
                          inputs=[scene_dropdown, model_dropdown, config_dropdown],
                          outputs=load_text)

        # ----------Exploration tab-------------------------------------------------------------------------------------
        # Initialize plot when selecting tab
        # explore_tab.select(update_explore_message, outputs=explore_message)
        # explore_tab.select(plot_3D_scene, inputs=plot_scene_options, outputs=scene_plot_output)
        # explore_tab.select(plot_coverage, outputs=coverage_plot_output)
        explore_tab.select(compute_next_camera_pose, outputs=explore_message)

        # Update plot when changing options
        plot_scene_options.change(plot_3D_scene, inputs=plot_scene_options, outputs=scene_plot_output)

        # Update camera, scene and plot when computing NBV
        nbv_button.click(compute_next_camera_pose, outputs=explore_message)
        # nbv_button.click(compute_next_camera_pose, inputs=plot_scene_options, outputs=scene_plot_output)
        # nbv_button.click(plot_coverage, outputs=coverage_plot_output)
        # nbv_button.click(update_frames, outputs=frame_images)

        explore_message.change(plot_3D_scene, inputs=plot_scene_options, outputs=scene_plot_output)
        explore_message.change(plot_coverage, outputs=coverage_plot_output)
        explore_message.change(update_frames, outputs=frame_images)
        explore_message.change(update_depths, outputs=depth_images)


    demo.launch(share=True)
