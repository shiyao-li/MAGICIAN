import os
from argparse import ArgumentParser
import utils
import utils_colmap
import helper
import json
import numpy as np
import torch
# import pycolmap
import trimesh
from pytorch3d.renderer import FoVPerspectiveCameras
import glob
from PIL import Image

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scene_dir", type=str)
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument("--pc_json_path", type=str, default=None)
    
    # Camera parameters
    parser.add_argument("--zfar", type=float, default=750)
    parser.add_argument("--fov", type=float, default=60)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=456)
    
    # Scene normalization
    parser.add_argument("--normalization_divider", type=float, default=20.0)
    
    args = parser.parse_args()

    ZFAR = args.zfar
    FOV = args.fov
    height = args.height
    width = args.width

    scene_dir = args.scene_dir
    if args.json_path is None:
        args.json_path = os.path.join(scene_dir, "test_gs.json")
    if args.pc_json_path is None:
        args.pc_json_path = os.path.join(scene_dir, "test_gs_pc.json")
    
    # with open(os.path.join(scene_dir, "test_gs/test_gs.json"), "r") as f:
    with open(args.json_path, "r") as f:
        data = json.load(f)
        try:
            data = data['neuschwanstein']['0']
        except:
            pass

    # with open(os.path.join(scene_dir, "test_gs/test_gs_pc.json"), "r") as f:
    with open(args.pc_json_path, "r") as f:
        pc_data = json.load(f)

    image_folder = os.path.join(scene_dir, "images")
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    Ts = torch.tensor(data['X_cam_history'])
    Rs = torch.tensor(data['V_cam_history'])
    points_3d = np.array(pc_data["points"]) # P, 3
    points_rgb = (np.array(pc_data["colors"]) * 255).astype(np.uint8) # P, 3

    # Filter out all-white images
    print("Checking for all-white images...")
    valid_indices = []
    removed_count = 0
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        img_array = np.array(img)

        # Check if image is all white (all pixels are 255)
        is_all_white = np.all(img_array == 255)

        if is_all_white:
            print(f"Removing all-white image: {os.path.basename(img_path)}")
            removed_count += 1
        else:
            valid_indices.append(i)

    print(f"Removed {removed_count} all-white images, keeping {len(valid_indices)} images")

    # Filter image_paths and camera parameters
    image_paths = [image_paths[i] for i in valid_indices]
    Ts = Ts[valid_indices]
    Rs = Rs[valid_indices]

    # Scene normalization
    print(f"Normalizing scene with divider {args.normalization_divider}")
    Ts = Ts / args.normalization_divider
    points_3d = points_3d / args.normalization_divider
    # End of scene normalization

    N_frames = Ts.shape[0]
    extrinsic = np.empty((N_frames, 3, 4))
    intrinsic = np.empty((N_frames, 3, 3))
    device = "cpu" # "cuda" if torch.cuda.is_available() else 
    for i in range(N_frames):
        sample_X_cam = Ts[i].view(1, 3)
        sample_V_cam = Rs[i].view(1, 2)
        R_sample, T_sample = utils.get_camera_RT(sample_X_cam, sample_V_cam)
        sample_camera = FoVPerspectiveCameras(R=R_sample, T=T_sample, zfar=ZFAR, device=device)
        K_matrix = sample_camera.get_projection_transform().get_matrix().transpose(-1, -2)
        sample_camera.K = K_matrix
        gs_camera = utils.convert_camera_from_pytorch3d_to_gs(sample_camera, height, width, device)[0]

        # extrinsic[i, :3, :3] = R_sample.numpy()
        # extrinsic[i, :3, 3] = T_sample.numpy()
        extrinsic[i, :3, :3] = gs_camera.R.cpu().numpy().T
        extrinsic[i, :3, 3] = gs_camera.T.cpu().numpy()

        fx = helper.fov2focal(gs_camera.FoVx, width)
        fy = helper.fov2focal(gs_camera.FoVy, height)
        # fx = helper.fov2focal(FOV, width)
        # fy = helper.fov2focal(FOV, height)
        cx = width / 2
        cy = height / 2
        intrinsic[i] = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

    camera_type = "PINHOLE"
    image_size = np.array([width, height])

    points_xyf = helper.create_pixel_coordinate_grid(N_frames, height, width)
    reconstruction = utils_colmap.batch_np_matrix_to_pycolmap_wo_track(
                points_3d,
                points_xyf,
                points_rgb,
                extrinsic,
                intrinsic,
                image_size,
                shared_camera=False,
                camera_type=camera_type,
            )
    reconstruction = utils_colmap.rename_colmap_recons_and_rescale_camera(reconstruction, image_paths)
    print(len(reconstruction.cameras), len(reconstruction.images), len(reconstruction.points3D))
    sparse_reconstruction_dir = os.path.join(scene_dir, "sparse/0")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)
    reconstruction.write_text(sparse_reconstruction_dir)
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(scene_dir, "sparse/0/points.ply"))
