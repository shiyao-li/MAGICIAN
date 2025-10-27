import numpy as np
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from pytorch3d.datasets import (
    collate_batched_meshes,
    render_cubified_voxels,
)
from pytorch3d.renderer import (
    TexturesVertex,
    TexturesAtlas,
)
from pytorch3d.transforms import(
    matrix_to_quaternion,
    quaternion_apply,
)
from .utils import *
import time

class CustomDataset(Dataset):

    def __init__(self, data_path, memory_threshold, rasterizer, screen_rasterizer, params, camera_axes, device,
                 save_to_json=False, load_from_json=False, json_name="models_list.json",
                 load_obj=True):
        self.data_path = data_path
        self.rasterizer = rasterizer
        self.screen_rasterizer = screen_rasterizer
        # self.renderer = renderer
        # self.embedder = embedder
        # self.preprocess = preprocess
        self.image_size = params.image_size
        self.camera_dist = params.camera_dist
        self.elevation = params.elevation
        self.azim_angle = params.azim_angle
        self.camera_axes = camera_axes
        self.side = params.side
        self.device = device

        self.load_obj = load_obj

        if not load_from_json:
            models = []
            for (dirpath, dirnames, filenames) in os.walk(data_path):
                for filename in filenames:
                    if filename[-4:] == ".obj":
                        models.append(os.path.join(dirpath, filename))
            models = remove_heavy_files(models, memory_threshold)
        else:
            with open(json_name) as f:
                dir_to_load = json.load(f)
            models = [os.path.join(data_path, path) for path in dir_to_load['models']]

        if save_to_json:
            dir_to_save = {}
            dir_to_save['models'] = [path[1+len(data_path):] for path in models]
            with open(json_name, 'w') as outfile:
                json.dump(dir_to_save, outfile)
            print("Saved models list in", json_name, ".")

        print("Database loaded.")
        self.models = models

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):  # -> Dict:

        model_path = self.models[idx]
        model = {}

        if self.load_obj:
            verts, faces, aux = load_obj(
                model_path,
                device=self.device,
                load_textures=False,
                create_texture_atlas=True,
                # texture_atlas_size=4,
                # texture_wrap="repeat",
            )

            verts = adjust_mesh(verts)

            # Create a textures object
            atlas = aux.texture_atlas

            model["verts"] = verts
            model["faces"] = faces.verts_idx
            model["textures"] = aux[4]
            model["atlas"] = aux.texture_atlas

        model["path"] = model_path
        return model

class CustomShapenetDataset(Dataset):

    def __init__(self, data_path, memory_threshold,
                 save_to_json=False, load_from_json=False, json_name="models_list.json",
                 official_split=False, adjust_diagonally=False,
                 load_obj=True):
        self.data_path = data_path
        self.official_split = official_split
        self.adjust_diagonally = adjust_diagonally

        self.load_obj = load_obj

        if not load_from_json:
            models = []
            for (dirpath, dirnames, filenames) in os.walk(data_path):
                for filename in filenames:
                    if filename[-4:] == ".obj":
                        models.append(os.path.join(dirpath, filename))
            models = remove_heavy_files(models, memory_threshold)
        else:
            with open(json_name) as f:
                dir_to_load = json.load(f)
            models = [os.path.join(data_path, path) for path in dir_to_load['models']]

        if save_to_json:
            dir_to_save = {}
            dir_to_save['models'] = [path[1+len(data_path):] for path in models]
            with open(json_name, 'w') as outfile:
                json.dump(dir_to_save, outfile)
            print("Saved models list in", json_name, ".")

        print("Database loaded.")
        self.models = models

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):  # -> Dict:

        model_path = self.models[idx]
        model = {}

        if self.load_obj:
            verts, faces, aux = load_obj(
                model_path,
                # device=self.device,
                load_textures=False,
                create_texture_atlas=True,
                # texture_atlas_size=4,
                # texture_wrap="repeat",
            )

            if self.adjust_diagonally:
                verts = adjust_mesh_diagonally(verts, diag_range=1.0)
            else:
                verts = adjust_mesh(verts)

            # Create a textures object
            atlas = aux.texture_atlas

            model["verts"] = verts
            model["faces"] = faces.verts_idx
            model["textures"] = aux[4]
            model["atlas"] = aux.texture_atlas

        model["path"] = model_path
        return model


class RGBDataset(Dataset):
    def __init__(self, data_path, alpha_max, use_future_images, scene_names=None,
                 frames_to_remove_json='frames_to_remove.pt'):
        self.data_path = data_path
        self.alpha_max = alpha_max
        self.use_future_images = use_future_images

        self.data = {}
        self.indices = {}
        current_idx = 0

        self.data['scenes'] = {}
        self.data['n_scenes'] = 0

        # If no scene name is provided, we just take all scenes in the folder
        if scene_names is None:
            # scene_names = os.listdir(self.data_path)
            scene_names = [scene_name for scene_name in os.listdir(self.data_path)
                           if os.path.isdir(os.path.join(self.data_path, scene_name))]

        if frames_to_remove_json in scene_names:
            scene_names.remove(frames_to_remove_json)

        self.frames_to_remove = torch.load(os.path.join(data_path, frames_to_remove_json))

        # For every scene...
        for scene_name in scene_names:
            self.data['n_scenes'] += 1
            self.data['scenes'][scene_name] = {}
            scene_path = os.path.join(self.data_path, scene_name)
            scene_path = os.path.join(scene_path, 'images')
            # print(scene_path)

            self.data['scenes'][scene_name]['trajectories'] = {}
            self.data['scenes'][scene_name]['n_trajectories'] = 0
            # ...And every trajectory...
            for trajectory_nb in os.listdir(scene_path):
                self.data['scenes'][scene_name]['n_trajectories'] += 1
                self.data['scenes'][scene_name]['trajectories'][trajectory_nb] = {}
                trajectory_path = os.path.join(scene_path, trajectory_nb)
                # print(trajectory_path)

                traj_length = len(os.listdir(trajectory_path))

                self.data['scenes'][scene_name]['trajectories'][trajectory_nb]['frames'] = {}
                self.data['scenes'][scene_name]['trajectories'][trajectory_nb]['n_frames'] = 0
                # ...We add all frames respecting conditions on the number of past (and, if required, future) frames
                for frame_name in os.listdir(trajectory_path):
                    frame_nb = frame_name[:-3]
                    short_path = scene_name + "/images/" + str(trajectory_nb) + "/" + str(frame_nb) + ".pt"

                    save_index = False
                    index_to_save = None

                    if int(frame_nb) >= self.alpha_max and (
                            (not self.use_future_images) or
                            int(frame_nb) < traj_length - self.alpha_max
                    ):
                        if not (short_path in self.frames_to_remove.keys()):
                            self.indices[str(current_idx)] = {'scene_name': scene_name,
                                                              'trajectory_nb': trajectory_nb,
                                                              'frame_nb': frame_nb}
                            save_index = True
                            index_to_save = current_idx
                            current_idx += 1

                    self.data['scenes'][scene_name]['trajectories'][trajectory_nb]['n_frames'] += 1
                    self.data['scenes'][scene_name][
                        'trajectories'][trajectory_nb][
                        'frames'][frame_nb] = {}
                    self.data['scenes'][scene_name][
                        'trajectories'][trajectory_nb][
                        'frames'][frame_nb][
                        'path'] = os.path.join(trajectory_path, str(frame_nb) + '.pt')
                    if save_index:
                        self.data['scenes'][scene_name][
                            'trajectories'][trajectory_nb][
                            'frames'][frame_nb][
                            'index'] = index_to_save

        print("Database loaded.")

    def __len__(self):
        # total_length = 0
        # for scene_name in self.data['scenes']:
        #     for trajectory_nb in self.data['scenes'][scene_name]['trajectories']:
        #         total_length += len(self.data['scenes'][scene_name]['trajectories'][trajectory_nb]['frames'].keys())
        # return total_length

        return len(self.indices.keys())

    def __getitem__(self, idx):  # -> Dict:

        scene_name = self.indices[str(idx)]['scene_name']
        trajectory_nb = self.indices[str(idx)]['trajectory_nb']
        frame_nb = self.indices[str(idx)]['frame_nb']

        frame_path = self.data['scenes'][scene_name]['trajectories'][trajectory_nb]['frames'][frame_nb]['path']
        frame = torch.load(frame_path, map_location='cpu')

        frame['path'] = frame_path
        frame['index'] = idx

        return frame

    def get_neighbor_frame(self, frame, alpha, device='cpu'):
        """

        :param frame: dictionary
        :param alpha: int
        :param device:
        :return:
        """
        idx = frame['index']
        scene_name = self.indices[str(idx)]['scene_name']
        trajectory_nb = self.indices[str(idx)]['trajectory_nb']
        frame_nb = str(int(self.indices[str(idx)]['frame_nb']) + alpha)

        neighbor_frame_path = self.data['scenes'][scene_name]['trajectories'][trajectory_nb]['frames'][frame_nb]['path']
        neighbor_frame = torch.load(neighbor_frame_path, map_location=device)

        neighbor_frame['path'] = neighbor_frame_path
        neighbor_frame['index'] = idx

        return neighbor_frame

    def get_neighbor_frame_from_idx(self, idx, alpha, device='cpu'):
        """

        :param idx: int
        :param alpha: int
        :param device:
        :return:
        """
        scene_name = self.indices[str(idx)]['scene_name']
        trajectory_nb = self.indices[str(idx)]['trajectory_nb']
        frame_nb = str(int(self.indices[str(idx)]['frame_nb']) + alpha)

        neighbor_frame_path = self.data['scenes'][scene_name]['trajectories'][trajectory_nb]['frames'][frame_nb]['path']
        neighbor_frame = torch.load(neighbor_frame_path, map_location=device)

        neighbor_frame['path'] = neighbor_frame_path
        neighbor_frame['index'] = idx

        return neighbor_frame


class SceneDataset(Dataset):
    def __init__(self, data_path, scene_names=None,
                 use_occupied_pose=True):
        self.data_path = data_path
        self.use_occupied_pose = use_occupied_pose

        # If no scene name is provided, we just take all scenes in the folder
        if scene_names is None:
            # scene_names = os.listdir(self.data_path)
            scene_names = [scene_name for scene_name in os.listdir(self.data_path)
                           if os.path.isdir(os.path.join(self.data_path, scene_name))]

        self.scene_names = scene_names

    def __len__(self):
        # total_length = 0
        # for scene_name in self.data['scenes']:
        #     for trajectory_nb in self.data['scenes'][scene_name]['trajectories']:
        #         total_length += len(self.data['scenes'][scene_name]['trajectories'][trajectory_nb]['frames'].keys())
        # return total_length

        return len(self.scene_names)

    def __getitem__(self, idx):  # -> Dict:

        scene_name = self.scene_names[idx]
        scene_path = os.path.join(self.data_path, scene_name)

        # Mesh info
        obj_name = scene_name + '.obj'
        for file_name in os.listdir(scene_path):
            if file_name[-4:] == '.obj':
                obj_name = file_name
                break

        # Settings info
        settings_file = os.path.join(scene_path, 'settings.json')
        with open(settings_file, "r") as read_content:
            settings = json.load(read_content)

        scene = {}
        scene['scene_name'] = scene_name
        scene['obj_name'] = obj_name
        scene['settings'] = settings

        # Info about occupied camera poses
        if self.use_occupied_pose:
            occupied_pose = torch.load(os.path.join(scene_path, 'occupied_pose.pt'), map_location=torch.device('cpu'))
            scene['occupied_pose'] = occupied_pose

        return scene