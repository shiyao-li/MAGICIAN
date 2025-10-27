# Description of config files

Below is a detailed description of all the hyperparameters involved in pretraining a surface coverage gain module.<br>
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
| `gt_max_diagonal` | float | Maximum diagonal length of the bounding box of ground truth meshes. |
| `n_points_surface` | int | Number of ground truth surface points for each mesh. |
| `surface_epsilon` | float | Distance threshold used in ground truth surface coverage computation. |

## 2. General training parameters

| Parameter | class | Description |
| :----- | :-----: | :----- |
| `start_from_scratch` | bool | If True, starts training of a new model. If False, resumes training from a previous checkpoint. |
| `total_batch_size` | int | Total batch size.|
| `batch_size` | int | Batch size per GPU.|
| `epochs` | int | Number of epochs. |
| `learning_rate` | float | Learning rate. |
| `schedule_learning_rate` | bool | If True, applies a scheduling strategy for learning rate. |
| `lr_factor` | float | Factor to scale the learning rate when using a scheduling strategy. |
| `lr_epochs` | List of int | Scales `learning_rate` by `lr_factor` for each epoch in the list. |
| `warmup` | int | Number of warmup iterations. |

## 3. Monitoring

| Parameter | class | Description |
| :----- | :-----: | :----- |
| `anomaly_detection` | bool | If True, uses anomaly detection from PyTorch. |
| `empty_cache_every_n_batch` | int | Number of batches to process before emptying cache, in order to avoid memory issues. |
| `check_gradients` | bool | If True, regularly prints the norm of gradients when updating the weights of neural modules. |

## 4. Occupancy probability module

| Parameter | class | Description |
| :----- | :-----: | :----- |
| `scone_occ_model_name` | str | Name of the weights file to use for the occupancy probability module. |
| `max_points_per_scone_occ_pass` | int | Maximum number of 3D points to process in a single forward pass using the occupancy probability module.|

## 5. Surface coverage gain module

| Parameter | class | Description |
| :----- | :-----: | :----- |
| `seq_len` | int | Number of points in the sequence processed by the self-attention unit of the surface coverage gain. |
| `n_proxy_points` | int | Number of 3D points used as proxies to represent the volumetric occupancy field. |
| `use_occ_to_sample_proxy_points` | bool | If True, uses the predicted occupancy probability field to sample points when computing surface coverage gain, as expected with a Monte Carlo integral. Should be True. |
| `min_occ_for_proxy_points` | float | Occupancy probability threshold. Points with probabilities under this threshold are considered to be empty, and won't participate in the surface coverage gain. |

## 6. Training parameters

| Parameter | Type | Description |
| :----- | :-----: | :----- |
| `n_view_max` | int | Maximum number of initial views, from which the model predicts the NBV. |
| `n_view_min` | int | Minimum number of initial views, from which the model predicts the NBV. |
| `camera_dist` | float | Distance between the camera poses and the origin of the world space. |
| `pole_cameras` | bool | If True, adds camera poses located at the poles to the list of camera candidates. |
| `n_camera_elev` | int | Number of different elevation values for candidate camera poses. |
| `n_camera_azim` | int | Number of different azimuth values for candidate camera poses. |
| `n_camera` | int | Total number of candidate camera poses. |
| `nbv_validation` | bool | If True, applies an additional validation process during training where only the quality of the predicted NBV is evaluated, rather than the distribution of surface coverage gains for all cameras. |