# Description of config files

Below is a detailed description of all the hyperparameters involved in training a complete MACARONS model.<br>
We organize the hyperparameters into different categories, as presented in the configuration files.

## 0. GPU management

| Parameter | Type | Description |
| :----- | :-----: | :----- |
| `ddp` | bool  | If True, distributed data parallel will be used. |
| `jz` | bool | Should be False. This parameter is used internally for Slurm configuration. |
| `CUDA_VISIBLE_DEVICES` | str | GPU devices to use during training (e.g. "0, 1, 2, 3"). |
| `WORLD_SIZE` | int | Number of GPU devices to use during training. |

## 1. Data

| Parameter | class | Description |
| :----- | :-----: | :----- |
| `data_path` | str | Path to the data directory. |
| `train_scenes` | List of str | List of scenes to explore training. Strings should be equal to the scene directory names in the dataset folder. |
| `data_augmentation` | bool | If True, will perform data augmentation during training. |
| `jitter_probability` | float | Probability to apply color jitter on images during training. Used only when `data_augmentation` is True. |
| `brightness_jitter_range` | float | Range for brightness jitter. |
| `contrast_jitter_range` | float | Range for contrast jitter. |
| `saturation_jitter_range` | float | Range for saturation jitter. |
| `hue_jitter_range` | float | Range for hue jitter. |
| `symmetry_probability` | float | Probability to symmetrize the mesh along any specified axis when loading a mesh of a scene during training. Used only when `data_augmentation` is True.|
| `axis_to_mirror` | List of int | List of axis along which meshes can be symmetrized. The list should contain 0, 1 and/or 2. |
| `scene_scale_factor` | float | Factor to scale the mesh of the scene during training. |

## 2. General training parameters

| Parameter | class | Description |
| :----- | :-----: | :----- |
| `start_from_scratch` | bool | If True, starts a new training from an initialized model. If False, resumes training from a previous checkpoint. |
| `online_learning` | bool | If False, freezes all weights and simply performs exploration and reconstruction. |
| `pretrained_model_path` | str | Name of the initialized model to load at the start of the training. |
| `batch_size` | int | Batch size per GPU. Should be equal to 1, since we suggest exploring 1 scene at a time per GPU. |
| `total_batch_size` | int | Total batch size, which represents the number of scenes you want to explore simultaneously. Should be equal to the total number of GPU.|
| `epochs` | int | Number of epochs. |
| `depth_learning_rate` | float | Learning rate for the depth module. |
| `scone_learning_rate` | float | Learning rate for both the occupancy probability module and the surface coverage gain module. |
| `schedule_learning_rate` | bool | If True, applies a scheduling strategy for learning rates. |
| `lr_factor` | float | Factor to scale the learning rates when using a scheduling strategy. |
| `depth_lr_epochs` | List of int | Scales `depth_learning_rate` by `lr_factor` for each epoch in the list. |
| `scone_lr_epochs` | List of int | Scales `scone_learning_rate` by `lr_factor` for each epoch in the list. |
| `depth_warmup` | int | Number of warmup iterations for depth module. |
| `scone_warmup` | int | Number of warmup iterations for both the occupancy probability module and the surface coverage gain module. |
| `warmup_phase` | int | Number of epochs during which we apply an even stronger warmup strategy. |
| `save_model_every_n_epoch` | int | Saves a checkpoint of the model every n epochs. |

## 3. Monitoring

| Parameter | class | Description |
| :----- | :-----: | :----- |
| `compute_time` | bool | If True, regularly prints the computation time taken by training iterations. |
| `check_gradients` | bool | If True, regularly prints the norm of gradients when updating the weights of neural modules. |
| `anomaly_detection` | bool | If True, uses anomaly detection from PyTorch. |
| `empty_cache_every_n_batch` | int | Number of batches to process before emptying cache, in order to avoid memory issues. |
| `reload_previous_model_if_bad_loss` | bool | If True, reloads the previously saved model when a large loss is computed during training. |
| `loss_peak_threshold` | float | Threshold used for reloading previous model when `reload_previous_model_if_bad_loss` is True. |

## 4. Camera management

| Parameter | class | Description |
| :----- | :-----: | :----- |
| `image_height` | int | Height of RGB images in pixels. |
| `image_width` | int | Width of RGB images in pixels. |
| `ambient_light_intensity` | float | Ambient light intensity in the scenes to explore. |
| `gathering_factor` | float | Proportion of pixels in the RGB images that will be back-projected into 3D to generate surface point clouds. |
| `sensor_range` | float | Range of the camera sensor. We evaluate surface coverage gains of points in the field of view of a camera only if they are located within this range. |
| `n_interpolation_steps` | int | Number of RGB images to capture along the way between the current camera pose and the next camera pose (the NBV). |
| `n_poses_in_trajectory` | int | Number of camera poses to visit before stopping the trajectory. |

## 5. Scene management

| Parameter | class | Description |
| :----- | :-----: | :----- |
| `n_proxy_points` | int | Number of 3D points used as proxies to represent the volumetric occupancy field. These points are uniformly sampled in the bounding box of the scene. |
| `score_threshold` | float | Threshold used for our score-based carving operation. We provide details in the appendix of the paper. |
| `carving_tolerance` | float | Tolerance threshold used for our carving operation. We provide details in the appendix of the paper. |
| `surface_cell_capacity` | int | Maximum number of surface points in each cell. We provide details in the paper. |
| `recompute_surface_every_n_loop` | int | Number of camera poses to compute before cleaning the reconstructed surface in order to remove noisy points. |
| `progressive_fill` | bool | If True, uses a slower but better algorithm to clean the reconstructed surface points. This method processes 3D surface points in small batches. |
| `max_points_per_progressive_fill` | int | Number of points in the small batches when applying `progressive_fill`. |
| `n_gt_surface_points` | int | Maximum number of ground truth surface points used to compute ground truth surface coverage gains. Used for evaluation only. |
| `distance_factor_th` | float | Threshold used for penalizing the distance to the camera when predicting surface coverage gains. We provide details in the main paper. |
| `remap_every_n_poses` | int | Number of poses to visit before recomputing a whole mapping of the scene. Useful at the end of trajectory, as it allows for saving a better mapping in the memory. |

## 6. Depth module

| Parameter | class | Description |
| :----- | :-----: | :----- |
| `use_perfect_depth` | bool | If True, uses perfect depth maps rather than predicted depth maps. Should be False. |
| `height` | int | Height of RGB images in pixels. |
| `width` | int | Width of RGB images in pixels. |
| `znear` | float | Depth of the closest depth plane when computing cost-volume features. |
| `zfar` | float | Depth of the farthest depth plane when computing cost-volume features. |
| `n_depth` | int | Number of depth planes to use when computing cost-volume features. |
| `n_alpha` | int | Number of successive frames to use when performing warping operations with depth planes in order to predict a depth map. |
| `n_apha_for_supervision` | int | Number of successive frames to use when performing warping operations with predicted depth maps in order to compute the reconstruction loss. |
| `use_future_frame_for_supervision` | bool | If True, also uses frame t+1 when performing warping operations to supervise the prediction of depth map t. |
| `alphas` | List of int | All values of k. Will use frames t+k when performing warping operations. |
| `regularity_loss` | bool | If True, applies an additional regularity term in the reconstruction loss. |
| `regularity_factor` | float | Weight for the regularity term in the reconstruction loss. |

## 7. Occupancy probability and Surface coverage gain modules (inspired by SCONE)

| Parameter | class | Description |
| :----- | :-----: | :----- |
| `prediction_neighborhood_size` | int  | Number of neighbor cells to use when computing local neighborhood features with occupancy probability module.  |
| `n_proxy_point_for_occupancy_supervision` | int | Number of 3D points in the batch when computing loss for the occupancy probability module. |
| `min_occ_for_proxy_points` | float | Occupancy probability threshold. Points with probabilities under this threshold are considered to be empty, and won't participate in the surface coverage gain. |
| `use_occ_to_sample_proxy_points` | bool | If True, uses the predicted occupancy probability field to sample points when computing surface coverage gain, as expected with a Monte Carlo integral. Should be True. |
| `seq_len` | int | Number of points in the sequence processed by the self-attention unit of the surface coverage gain. |
| `n_freeze_epochs` | int | Number of epochs during which we freeze the weights of the occupancy probability and surface coverage gain modules at the start of the training. Can be greater than 0 if we use pretrained weights.|

## 8. Memory replay

| Parameter | class | Description |
| :----- | :-----: | :----- |
| `memory_dir_name` | str | Name of the memory directory. |
| `n_memory_trajectories` | int | Number of trajectories for which we store data into the memory. We erase data from older trajectories to register data from new ones. |
| `n_memory_loops` | int | Number of memory replay iterations to perform with the depth module. |
| `n_memory_samples` | int | Number of depth maps to predict (batch size) during each memory replay iteration with the depth module. |
| `n_memory_scene_loops` | int | Number of memory replay iterations to perform with both the occupancy probability module and the surface coverage gains module. |
| `save_depth_every_n_frame` | int | Every n poses, we store the predicted depth map into the memory. |
| `n_max_memory_depths_for_partial_pc` | int | Maximum number of depth maps to load from the memory when reconstructing a surface point cloud during a memory replay iteration. |
| `random_poses_in_memory_scene_loops` | bool | If True, loads RGB images from random camera poses when performing memory replay iterations. If False, loads RGB images from successive camera poses along a trajectory. |
| `n_poses_in_memory_scene_loops` | int | Number of camera poses for which we predict a surface coverage gain when we perform a memory replay iteration (batch size for the surface coverage gain module). |
