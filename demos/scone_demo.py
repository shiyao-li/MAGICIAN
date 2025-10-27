import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
import torch

sys.path.append(os.path.abspath('../'))
from macarons.testers.shapenet import *
from macarons.utility.render_utils import plot_point_cloud, plot_graph


pc_size = 2048
n_view_max = 10
test_novel = False
test_number = -1

start_object = 3531
# start_pose = 32

# Path to ShapeNetCore.v1 folder
data_path = "../data/ShapeNetCore.v1"

object_folder = data_path
occupancy_model_folder = "../weights/scone/occupancy"
coverage_model_folder = "../weights/scone/coverage_gain"
config_folder = "../configs/scone/coverage_gain"
numGPU = 0
selectable_paths = False

n_3D_points = 20000
point_size = 3
# n_frames_to_show = 4
# frame_shape = (456, 256)

# test_resolution = 0.05


class GlobalParameters:
    def __init__(self):
        # Selection
        self.object_folder = object_folder
        self.occupancy_model_folder = occupancy_model_folder
        self.coverage_model_folder = coverage_model_folder
        self.config_folder = config_folder

        # self.object_list = os.listdir(self.object_folder)
        self.occupancy_model_list = os.listdir(self.occupancy_model_folder)
        self.coverage_model_list = os.listdir(self.coverage_model_folder)
        self.config_list = os.listdir(self.config_folder)

        self.object_path = ""
        self.occupancy_model_path = ""
        self.coverage_model_path = ""
        self.occupancy_model_name = ""
        self.coverage_model_name = ""
        self.config_path = ""

        self.object_idx = None

        self.loading_ok = False

        # Config parameters object
        self.params = None

        # Dataloader object
        self.dataloader = None
        self.dataset = None

        # Mesh object and data
        # self.mesh = None
        # self.obj_name = None
        # self.mesh_path = None
        self.mesh_dict = None
        self.part_pc = None
        self.coverage_matrix = None

        # Model objects
        self.scone_occ = None
        self.scone_vis = None

        # Camera object
        self.camera = None

        # Harmonics data
        self.base_harmonics = None
        self.h_polar = None
        self.h_azim = None

        # Reconstruction data
        self.X_world = None
        self.cov_pred = None
        self.cov_truth = None
        self.view_idx = None

        self.X_cam_world = None
        self.camera_dist = None
        self.camera_elev = None
        self.camera_azim = None

        self.proxy_points = None
        self.occ_prob = None
        self.vis_pred = None

        self.prediction_view_transform = None
        self.prediction_box_center = None
        self.X_cam = None

        self.pc = None
        self.gt_pc = None
        self.X = None

        # Plot state
        self.plot_ok = False
        self.occ_threshold = 0.5

        # Coverage metric
        self.coverage_evolution = None

        # Device
        self.device = None

        self.computation_done = False
        self.update_board = False


global_params = GlobalParameters()


def refresh_object_folder(new_folder):
    files_list = os.listdir(new_folder)
    global_params.object_folder = new_folder
    # return gr.Dropdown.update(choices=files_list)


def refresh_occupancy_model_folder(new_folder):
    files_list = os.listdir(new_folder)
    global_params.occupancy_model_folder = new_folder
    return gr.Dropdown.update(choices=files_list)


def refresh_coverage_model_folder(new_folder):
    files_list = os.listdir(new_folder)
    global_params.coverage_model_folder = new_folder
    return gr.Dropdown.update(choices=files_list)


def refresh_config_folder(new_folder):
    files_list = os.listdir(new_folder)
    global_params.config_folder = new_folder
    return gr.Dropdown.update(choices=files_list)


def load_selected(object_folder, occupancy_model_name, coverage_model_name, config_name):
    if min(len(object_folder), len(occupancy_model_name), len(coverage_model_name), len(config_name)) == 0:
        return "<p style='color:#dc2626'>" \
               "<b style='color:#dc2626'>Error:</b> Please select an object, models and a config file." \
               "</p>"

    else:
        # Update selection parameters
        global_params.occupancy_model_path = os.path.join(global_params.occupancy_model_folder, occupancy_model_name)
        global_params.coverage_model_path = os.path.join(global_params.coverage_model_folder, coverage_model_name)
        global_params.occupancy_model_name = occupancy_model_name
        global_params.coverage_model_name = coverage_model_name
        global_params.config_path = os.path.join(global_params.config_folder, config_name)

        # Load parameters from config file
        global_params.params = load_params(global_params.config_path)

        global_params.params.jz = False
        global_params.params.ddp = False
        global_params.params.numGPU = numGPU
        global_params.params.WORLD_SIZE = 1
        global_params.params.batch_size = 1
        global_params.params.total_batch_size = 1
        global_params.params.true_monte_carlo_sampling = True  # todo: not necessary

        # Setup device
        # device = 'cpu'
        global_params.device = setup_device(global_params.params)

        # Create dataloader
        global_params.params.data_path = global_params.object_folder
        print("Data path:", global_params.object_folder)
        train_dataloader, _, test_dataloader = get_shapenet_dataloader(batch_size=global_params.params.batch_size,
                                                                       ddp=False, jz=False,
                                                                       world_size=None, ddp_rank=None,
                                                                       test_number=-1,
                                                                       test_novel=test_novel,
                                                                       load_obj=False,
                                                                       data_path=object_folder,
                                                                       shuffle=False)
        global_params.dataloader = train_dataloader  # todo: to change?
        global_params.dataset = global_params.dataloader.dataset

        # Load models
        global_params.scone_occ = load_scone_occ(global_params.params,
                                                 global_params.occupancy_model_name,
                                                 ddp_model=True,device=global_params.device)
        global_params.scone_vis = load_scone_vis(global_params.params,
                                                 global_params.coverage_model_name,
                                                 ddp_model=True, device=global_params.device)
        global_params.scone_occ.eval()
        global_params.scone_vis.eval()

        # Harmonics data
        global_params.base_harmonics, global_params.h_polar, global_params.h_azim = get_all_harmonics_under_degree(
            global_params.params.harmonic_degree,
            global_params.params.view_state_n_elev,
            global_params.params.view_state_n_azim,
            global_params.device)

        # Select a random object
        global_params.object_idx = np.random.randint(low=0, high=len(global_params.dataset))  # 52
        global_params.object_idx = start_object # 52
        # global_params.object_path = os.path.join(global_params.object_folder, object_name)

        global_params.loading_ok = True
        # load_object()

    return "Object " + str(global_params.object_idx) + " has been loaded for reconstruction."


def load_object():
    if global_params.loading_ok:
        clear_spherical_harmonics_cache()
        global_params.mesh_dict = global_params.dataset[global_params.object_idx]
        # global_params.mesh_dict['path'] = [global_params.mesh_dict['path']]

        global_params.object_path = global_params.mesh_dict['path']

        global_params.cov_pred = torch.zeros(0, global_params.params.n_camera, 1, device=global_params.device)
        global_params.cov_truth = torch.zeros(0, global_params.params.n_camera, 1, device=global_params.device)

        # ----------Load input mesh and ground truth data-----------------------------------------------------------
        global_params.coverage_evolution = []

        # Loading info about partial point clouds and coverages
        global_params.part_pc, global_params.coverage_matrix = get_gt_partial_point_clouds(path=global_params.object_path,
                                                                                           normalization_factor=1.,
                                                                                           device=global_params.device)

        # Initial dense sampling
        global_params.X_world = sample_X_in_box(x_range=global_params.params.gt_max_diagonal,
                                                n_sample=global_params.params.n_proxy_points,
                                                device=global_params.device)

        # ----------Set camera candidates for coverage prediction---------------------------------------------------
        global_params.X_cam_world, global_params.camera_dist, global_params.camera_elev, global_params.camera_azim = get_cameras_on_sphere(
            global_params.params, global_params.device, pole_cameras=global_params.params.pole_cameras)

        global_params.view_idx = torch.randperm(len(global_params.camera_elev), device=global_params.device)[:1]

        global_params.prediction_cam_idx = global_params.view_idx[0]
        global_params.prediction_box_center = torch.Tensor([0., 0., global_params.params.camera_dist]).to(global_params.device)

        # Move camera coordinates from world space to prediction view space, and normalize them for prediction box
        global_params.prediction_R, global_params.prediction_T = look_at_view_transform(
            dist=global_params.camera_dist[global_params.prediction_cam_idx],
            elev=global_params.camera_elev[global_params.prediction_cam_idx],
            azim=global_params.camera_azim[global_params.prediction_cam_idx],
            device=global_params.device)
        global_params.prediction_camera = FoVPerspectiveCameras(device=global_params.device,
                                                                R=global_params.prediction_R,
                                                                T=global_params.prediction_T)
        global_params.prediction_view_transform = global_params.prediction_camera.get_world_to_view_transform()

        global_params.X_cam = global_params.prediction_view_transform.transform_points(global_params.X_cam_world)
        global_params.X_cam = normalize_points_in_prediction_box(points=global_params.X_cam,
                                                                 prediction_box_center=global_params.prediction_box_center,
                                                                 prediction_box_diag=global_params.params.gt_max_diagonal)
        _, global_params.elev_cam, global_params.azim_cam = get_spherical_coords(global_params.X_cam)

        # GT pc
        gt_pc = torch.vstack([part_pc for part_pc in global_params.part_pc])
        gt_pc = gt_pc[torch.randperm(len(gt_pc))[:n_3D_points]]
        gt_pc = global_params.prediction_view_transform.transform_points(gt_pc)
        global_params.gt_pc = normalize_points_in_prediction_box(points=gt_pc,
                                                                 prediction_box_center=global_params.prediction_box_center,
                                                                 prediction_box_diag=global_params.params.gt_max_diagonal).view(1, -1, 3)

        return capture_pc_and_compute_nbv()

    else:
        return update_reconstruction_message()


def reload_object():
    global_params.object_idx = np.random.randint(low=0, high=len(global_params.dataset))  # 52
    global_params.loading_ok = True
    return load_object()


def capture_pc_and_compute_nbv():
    global_params.computation_done = False
    X_view = global_params.X_cam[global_params.view_idx]
    # X_cam = X_cam.view(1, params.n_camera, 3)

    # Compute current coverage
    current_coverage = compute_surface_coverage_from_cam_idx(global_params.coverage_matrix,
                                                     global_params.view_idx).detach().item()
    global_params.coverage_evolution.append(current_coverage)

    # ----------Capture initial observations----------------------------------------------------------------

    # Points observed in initial views
    pc = torch.vstack([global_params.part_pc[pc_idx] for pc_idx in global_params.view_idx])
    n_view = len(global_params.view_idx)
    print("N view:", n_view)

    # Downsampling partial point cloud
    # pc = pc[torch.randperm(len(pc))[:n_view * params.seq_len]]
    pc = pc[torch.randperm(len(pc))[:n_view * pc_size]]

    # Move partial point cloud from world space to prediction view space,
    # and normalize them in prediction box
    pc = global_params.prediction_view_transform.transform_points(pc)
    pc = normalize_points_in_prediction_box(points=pc,
                                            prediction_box_center=global_params.prediction_box_center,
                                            prediction_box_diag=global_params.params.gt_max_diagonal).view(1, -1, 3)
    global_params.pc = pc

    # Move proxy points from world space to prediction view space, and normalize them in prediction box
    X = global_params.prediction_view_transform.transform_points(global_params.X_world)
    X = normalize_points_in_prediction_box(points=X,
                                           prediction_box_center=global_params.prediction_box_center,
                                           prediction_box_diag=global_params.params.gt_max_diagonal
                                           )

    # Filter Proxy Points using pc shape from view cameras
    R_view, T_view = look_at_view_transform(eye=X_view,
                                            at=torch.zeros_like(X_view),
                                            device=global_params.device)
    view_cameras = FoVPerspectiveCameras(R=R_view, T=T_view, zfar=1000, device=global_params.device)
    X, _ = filter_proxy_points(view_cameras, X, pc.view(-1, 3), filter_tol=global_params.params.filter_tol)
    X = X.view(1, X.shape[0], 3)

    global_params.X = X

    # Compute view state vector and corresponding view harmonics
    view_state = compute_view_state(X, X_view,
                                    global_params.params.view_state_n_elev, global_params.params.view_state_n_azim)
    view_harmonics = compute_view_harmonics(view_state,
                                            global_params.base_harmonics, global_params.h_polar, global_params.h_azim,
                                            global_params.params.view_state_n_elev, global_params.params.view_state_n_azim)
    occ_view_harmonics = 0. + view_harmonics
    if global_params.params.occ_no_view_harmonics:
        occ_view_harmonics *= 0.
    if global_params.params.no_view_harmonics:
        view_harmonics *= 0.

    # Compute occupancy probabilities
    with torch.no_grad():
        global_params.occ_prob = compute_occupancy_probability(scone_occ=global_params.scone_occ,
                                                               pc=pc,
                                                               X=X,
                                                               view_harmonics=occ_view_harmonics,
                                                               max_points_per_pass=global_params.params.max_points_per_scone_occ_pass
                                                               ).view(-1, 1)

    global_params.proxy_points, view_harmonics, sample_idx = sample_proxy_points(X[0], global_params.occ_prob,
                                                                                 view_harmonics.squeeze(dim=0),
                                                                                 n_sample=global_params.params.seq_len,
                                                                                 min_occ=global_params.params.min_occ_for_proxy_points,
                                                                                 use_occ_to_sample=global_params.params.use_occ_to_sample_proxy_points,
                                                                                 return_index=True)

    global_params.proxy_points = torch.unsqueeze(global_params.proxy_points, dim=0)
    view_harmonics = torch.unsqueeze(view_harmonics, dim=0)

    # ----------Predict Coverage Gains------------------------------------------------------------------------------
    visibility_gain_harmonics = global_params.scone_vis(global_params.proxy_points, view_harmonics=view_harmonics)
    if global_params.params.true_monte_carlo_sampling:
        global_params.proxy_points = torch.unsqueeze(global_params.proxy_points[0][sample_idx], dim=0)
        visibility_gain_harmonics = torch.unsqueeze(visibility_gain_harmonics[0][sample_idx], dim=0)
        print("Applying true MC sampling.")

    with torch.no_grad():
        global_params.vis_pred = global_params.scone_vis.compute_visibilities(global_params.proxy_points,
                                                                visibility_gain_harmonics,
                                                                global_params.X_cam.view(1, -1, 3))
        global_params.cov_pred = global_params.scone_vis.compute_coverage_gain(global_params.proxy_points,
                                                                 visibility_gain_harmonics,
                                                                 global_params.X_cam.view(1, -1, 3)).view(-1, 1)

    # ----------Compute ground truth information scores----------
    global_params.cov_truth = compute_gt_coverage_gain_from_precomputed_matrices(coverage=global_params.coverage_matrix,
                                                                                 initial_cam_idx=global_params.view_idx)

    # ----------Identify maximum gain to get NBV camera----------
    (max_gain, max_idx) = torch.max(global_params.cov_pred, dim=0)

    # Add NBV camera
    global_params.view_idx = torch.cat((global_params.view_idx,
                                        torch.Tensor([max_idx]).long().to(global_params.device)), dim=0)

    global_params.computation_done = True
    return update_reconstruction_message()

def plot_3D_scene(checkboxes):
    """
    "GT surface",
    "Visited camera poses",
    "Scanned surface",
    "Occupancy field",
    "NBV camera",
    "Visibility gain for NBV"

    :param checkboxes:
    :return:
    """
    # X = torch.rand(20000, 3)
    X = torch.zeros(0, 3)
    c = torch.zeros(0, 3)

    global_params.plot_ok = False

    if "Scanned surface" in checkboxes:
        X = global_params.pc.view(-1, 3)
        c = (X - X.min()) / (X.max() - X.min())
        global_params.plot_ok = True

    if "Occupancy field" in checkboxes:
        X = global_params.X.view(-1, 3)
        magma_colors = torch.Tensor(plt.get_cmap('magma')(np.linspace(0, 1, 256))[..., :3]).to(global_params.device)
        c = magma_colors[(255 * global_params.occ_prob.view(-1).clamp(0., 1.).view(-1)).long()]

        occ_mask = global_params.occ_prob.view(-1) > global_params.occ_threshold
        X = X[occ_mask]
        c = c[occ_mask]
        global_params.plot_ok = True

    if "Visibility gain for NBV" in checkboxes:
        X = global_params.proxy_points.view(-1, 4)[..., :3]
        viridis_colors = torch.Tensor(plt.get_cmap('viridis')(np.linspace(0, 1, 256))[..., :3]).to(global_params.device)
        nbv_idx = global_params.view_idx[-1]
        c = global_params.vis_pred[0, nbv_idx]
        c = (c - c.min()) / (c.max() - c.min())
        c = viridis_colors[(255 * c.clamp(0., 1.).view(-1)).long()]

        # occ_mask = global_params.occ_prob.view(-1) > global_params.occ_threshold
        # X = X[occ_mask]
        # c = c[occ_mask]
        global_params.plot_ok = True

    if "GT surface" in checkboxes:
        X = global_params.gt_pc.view(-1, 3)
        c = (X - X.min()) / (X.max() - X.min())
        # c = 0.5 * torch.ones_like(X)
        global_params.plot_ok = True

    if not global_params.plot_ok:
        X = torch.zeros(1, 3, device=global_params.device)
        c = torch.ones(1, 3, device=global_params.device)

    if "Visited camera poses" in checkboxes:
        X_cam = global_params.X_cam[global_params.view_idx[:-1]].view(-1, 3)
        c_cam = torch.zeros_like(X_cam)
        c_cam[..., 0] += 1

        X_perm = torch.randperm(len(X))
        X = torch.cat((X_cam, X[X_perm][:n_3D_points - len(X_cam)]), dim=0)
        c = torch.cat((c_cam, c[X_perm][:n_3D_points - len(c_cam)]), dim=0)
        global_params.plot_ok = True

    if "NBV camera" in checkboxes:
        X_nbv = global_params.X_cam[global_params.view_idx[-1:]].view(-1, 3)
        c_nbv = torch.zeros_like(X_nbv)
        c_nbv[..., 1] += 1

        X_perm = torch.randperm(len(X))
        X = torch.cat((X_nbv, X[X_perm][:n_3D_points - len(X_nbv)]), dim=0)
        c = torch.cat((c_nbv, c[X_perm][:n_3D_points - len(c_nbv)]), dim=0)
        global_params.plot_ok = True

    return plot_point_cloud(X, c, name="3D Scene", max_points=n_3D_points, point_size=point_size,
                            width=730, height=550)  # height=716


def plot_coverage():
    x = np.arange(len(global_params.coverage_evolution))
    return plot_graph(x=x, y=np.array(global_params.coverage_evolution),
                      x_label="NBV iterations", y_label="Total surface coverage",
                      width=730, height=250)


def plot_proportionality():
    x = np.arange(len(global_params.cov_truth))
    y1 = (global_params.cov_pred * global_params.cov_truth.mean()
          / global_params.cov_pred.mean()).view(-1).cpu().numpy()
    y2 = global_params.cov_truth.view(-1).cpu().numpy()

    return plot_graph(x=x, y=[y1, y2],
                      x_label="Camera indices", y_label="Surface coverage gains",
                      names=["Normalized prediction", "Ground truth"],
                      width=730, height=250)


def update_scene_plot_message(checkboxes):
    """
    "GT surface",
    "Visited camera poses",
    "Scanned surface",
    "Occupancy field",
    "NBV camera",
    "Visibility gain for NBV"

    :param checkboxes:
    :return:
    """

    res0 = ""
    res1 = ""

    if "Scanned surface" in checkboxes:
        # res0 += "The 3D surface scanned by the depth sensor is plotted in multicolor."
        res0 = "This plot represents the 3D surface captured by the depth sensor from the previously " \
               "visited camera poses, visualized using multiple colors.\n"

    if "Occupancy field" in checkboxes:
        res0 = "This plot represents the occupancy probability field predicted by the model. " \
               "The brighter the points, the higher their probability.\n"

    if "Visibility gain for NBV" in checkboxes:
        res0 = "This plot represents, for each 3D point in the volume, the visibility gains predicted by the model " \
               "in direction of the next camera pose to be visited (the NBV). " \
               "The brighter the points, the higher their gain.\n" \
               "The NBV is selected as the pose with the highest surface coverage gain, " \
               "i.e. the highest aggregated visibility gains.\n"

    if "GT surface" in checkboxes:
        res0 = "This plot represents the ground truth 3D surface of the object, visualized using multiple colors.\n"

    if "Visited camera poses" in checkboxes:
        res1 += "The camera poses that have already been visited are displayed in red. "

    if "NBV camera" in checkboxes:
        res1 += "The next camera pose to be visited (the Next Best View or NBV) is highlighted in green."

    return res0 + res1


def update_reconstruction_message():
    res = ""
    global_params.update_board = False
    if not global_params.loading_ok:
        res += "<b style='color:#dc2626'>Error:</b> Please first select the data folder, models and a config file " \
               "in the <tt style='color:#ea580c'>Selection</tt> tab."

    elif global_params.computation_done:
        res += "Computation done for pose " + str(len(global_params.view_idx) - 1) + ". "
        res += "Total surface coverage has reached " + str(global_params.coverage_evolution[-1])
        global_params.update_board = True
        global_params.computation_done = False
    return res


# ----------Main--------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to launch a gradio demo showing how scone works.')
    parser.add_argument('-d', '--device', type=int, help='Number of the GPU device to use for the demo.')

    args = parser.parse_args()

    if args.device:
        numGPU = args.device

    print("Using GPU device " + str(numGPU) + '.')

    with gr.Blocks() as demo:
        # Header text
        gr.HTML("<center>"
                "<font size='6'> "
                    "<b>SCONE: Surface Coverage Optimization in Unknown Environments by Volumetric Integration</b>"
                "</font><br>"
                "<font size='5'>"
                    "NeurIPS 2022 (Spotlight)"
                "</font><br><br>"
                "<font size='4'> "
                    "<b>Antoine Guédon, Pascal Monasse, Vincent Lepetit</b><br>"
                    "LIGM, Ecole des Ponts, Univ Gustave Eiffel, CNRS"
                "</font>"
                )

        gr.HTML("Please select a 3D object, models and a config file using the <tt style='color:#ea580c'>Selection</tt> tab.<br>"
                "Then, use the <tt style='color:#ea580c'>Reconstruction</tt> tab to reconstruct the selected 3D object.")

        # First tab: Select a 3D object and model------------------------------------------------------------------------
        with gr.Tab("Selection"):

            # Select a 3D object
            gr.HTML("<h2>1. Select a 3D object</h2>")
            # if selectable_paths:
            gr.HTML(# "Path to 3D scene folder.<br>"
                    "<p style='color:#9ca3af'>"
                        "Enter the path to the ShapeNet folder, and press the Enter key."
                    "</p>"
                    #"<small style='color:#9ca3af'>Enter the path to the folder containing all 3D scenes subfolders.</small>"
                    )
            object_text_input = gr.Textbox(#label="Path to 3D scene folder.",
                                          #info="Enter the path to the folder containing all 3D scenes subfolders.",
                                          value=global_params.object_folder,
                                          placeholder="objects...",
                                          show_label=False)
            object_text_input.style(container=False)

            # gr.HTML("<p style='color:#9ca3af'>"
            #             "Please use the following dropdown list to select an object to reconstruct."
            #         "</p>")
            # object_dropdown = gr.Dropdown(choices=global_params.object_list,
            #                              #label="Select a scene to explore.",
            #                              #info="Select a scene to explore.",
            #                              # value=global_params.scene_list[0],
            #                              max_choices=1, multiselect=False, interactive=True, show_label=False)
            # object_dropdown.style(container=False)

            # Select a SconeOcc model
            gr.HTML(#"<br>"
                    "<h2>2. Select an occupancy model</h2>")
            if selectable_paths:
                gr.HTML("<p style='color:#9ca3af'>"
                            "Enter the path to the folder containing all occupancy models, and press the Enter key.<br>"
                            "Models should have .pth extension."
                        "</p>")
                occupancy_model_text_input = gr.Textbox(
                    value=global_params.occupancy_model_folder,
                    placeholder="occupancy models...",
                    show_label=False)
                occupancy_model_text_input.style(container=False)

            gr.HTML("<p style='color:#9ca3af'>"
                    "Please use the following dropdown list to select an occupancy model for reconstruction."
                    "</p>")
            occupancy_model_dropdown = gr.Dropdown(choices=global_params.occupancy_model_list,
                                         # info="Select a model for exploration.",
                                         max_choices=1, multiselect=False, interactive=True, show_label=False)
            occupancy_model_dropdown.style(container=False)

            # Select a SconeVis model
            gr.HTML(#"<br>"
                    "<h2>2. Select a coverage gain model</h2>")
            if selectable_paths:
                gr.HTML("<p style='color:#9ca3af'>"
                            "Enter the path to the folder containing all coverage gain models, and press the Enter key.<br>"
                            "Models should have .pth extension."
                        "</p>")
                coverage_model_text_input = gr.Textbox(
                    value=global_params.coverage_model_folder,
                    placeholder="coverage gain models...",
                    show_label=False)
                coverage_model_text_input.style(container=False)

            gr.HTML("<p style='color:#9ca3af'>"
                    "Please use the following dropdown list to select a coverage gain model for reconstruction."
                    "</p>")
            coverage_model_dropdown = gr.Dropdown(choices=global_params.coverage_model_list,
                                         # info="Select a model for exploration.",
                                         max_choices=1, multiselect=False, interactive=True, show_label=False)
            coverage_model_dropdown.style(container=False)

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

        # Second tab: Explore a 3D object--------------------------------------------------------------------------------
        reconstruction_tab = gr.Tab("Reconstruction")
        with reconstruction_tab:
            with gr.Row():
                #image_input = gr.Image()
                with gr.Column():
                    # gr.HTML("Plot either the GT surface points or the reconstructed surface points.")
                    restart_button = gr.Button("Restart with another random object")

                    # Plot options
                    plot_scene_options = gr.CheckboxGroup(choices=["Visited camera poses",
                                                                   "Scanned surface",
                                                                   "Occupancy field",
                                                                   "NBV camera",
                                                                   "Visibility gain for NBV",
                                                                   "GT surface",
                                                                   ],
                                                          value=["Visited camera poses",
                                                                 "Scanned surface",
                                                                 "NBV camera"],
                                                          info="Elements to display.",
                                                          interactive=True,
                                                          show_label=False
                                                          )
                    plot_scene_options.style()
                    # gr.HTML("Plot the computed trajectory.")

                    # Plots
                    # with gr.Row():
                    coverage_plot_output = gr.Plot()
                    proportionality_plot_output = gr.Plot()

                # image_output = gr.Image()
                with gr.Column():
                    scene_plot_output = gr.Plot()
                    scene_plot_text = gr.TextArea(lines=5, show_label=False)
            reconstruction_message = gr.HTML("")
            nbv_button = gr.Button("Compute next camera pose")

        # with gr.Accordion("3D scenes"):
        #     gr.Markdown("To do...")

        # ----------Selection tab---------------------------------------------------------------------------------------
        if selectable_paths:
            object_text_input.submit(refresh_object_folder,
                                     inputs=object_text_input)
            occupancy_model_text_input.submit(refresh_occupancy_model_folder,
                                              inputs=occupancy_model_text_input,
                                              outputs=occupancy_model_dropdown)
            coverage_model_text_input.submit(refresh_coverage_model_folder,
                                             inputs=coverage_model_text_input,
                                             outputs=coverage_model_dropdown)
            config_text_input.submit(refresh_config_folder,
                                     inputs=config_text_input,
                                     outputs=config_dropdown)

        load_button.click(load_selected,
                          inputs=[object_text_input, occupancy_model_dropdown, coverage_model_dropdown, config_dropdown],
                          outputs=load_text)

        # ----------Exploration tab-------------------------------------------------------------------------------------
        # Initialize plot when selecting tab
        reconstruction_tab.select(load_object, outputs=reconstruction_message)
        # reconstruction_tab.select(plot_coverage, outputs=coverage_plot_output)
        #
        # explore_tab.select(update_explore_message, outputs=explore_message)
        #
        # # Update plot when changing options
        plot_scene_options.change(plot_3D_scene, inputs=plot_scene_options, outputs=scene_plot_output)
        plot_scene_options.change(update_scene_plot_message, inputs=plot_scene_options, outputs=scene_plot_text)
        #
        # # Update camera, scene and plot when computing NBV
        nbv_button.click(capture_pc_and_compute_nbv, outputs=reconstruction_message)
        # # nbv_button.click(compute_next_camera_pose, inputs=plot_scene_options, outputs=scene_plot_output)
        # # nbv_button.click(plot_coverage, outputs=coverage_plot_output)
        # # nbv_button.click(update_frames, outputs=frame_images)
        #
        reconstruction_message.change(plot_3D_scene, inputs=plot_scene_options, outputs=scene_plot_output)
        reconstruction_message.change(update_scene_plot_message, inputs=plot_scene_options, outputs=scene_plot_text)
        reconstruction_message.change(plot_coverage, outputs=coverage_plot_output)
        reconstruction_message.change(plot_proportionality, outputs=proportionality_plot_output)
        # explore_message.change(update_frames, outputs=frame_images)
        # explore_message.change(update_depths, outputs=depth_images)

        restart_button.click(reload_object, outputs=reconstruction_message)


    demo.launch(share=True)