# Description of config files

Below is a detailed description of all the hyperparameters involved in pretraining an occupancy probability module.<br>
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
| `n_sample` | str | Number of 3D points in the batch when computing the loss. |
| `seq_len` | str | Number of points in the sequence processed by the self-attention unit to compute the global feature. Not used to compute local neighborhood features.|

## 5. Training parameters

| Parameter | Type | Description |
| :----- | :-----: | :----- |
| `n_view_max` | int | Maximum number of initial views, from which the model predicts the occupancy probability field. |
| `n_view_min` | int | Minimum number of initial views, from which the model predicts the occupancy probability field. |
| `camera_dist` | float | Distance between the camera poses and the origin of the world space. |
| `pole_cameras` | bool | If True, adds camera poses located at the poles to the list of camera candidates. |
| `n_camera_elev` | int | Number of different elevation values for candidate camera poses. |
| `n_camera_azim` | int | Number of different azimuth values for candidate camera poses. |
| `n_camera` | int | Total number of candidate camera poses. |
