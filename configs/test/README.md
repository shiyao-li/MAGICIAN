# Description of config files

Below is a detailed description of all the hyperparameters involved in evaluating our models.<br>

## 1. Large-scale 3D scenes exploration and reconstruction with an RGB camera

| Parameter | Type | Description |
| :----- | :-----: | :----- |
| `numGPU` | int  | GPU device to use for evaluation. |
| `dataset_path` | str | Path to the data directory. |
| `test_scenes` | List of str | List of scenes to explore during evaluation. Strings should be equal to the scene directory names in the dataset folder. |
| `params_name` | str | Name of the config file corresponding to the model to be evaluated. |
| `model_name` | str | Name of the weights file corresponding to the model to be evaluated. |
| `results_json_name` | str | Name of the json file in which results will be saved. |
| `test_resolution` | str | Distance threshold used to compute surface coverage during evaluation. |
| `use_perfect_depth_map` | bool | If True, perfect depth maps will be used during evaluation, rather than predicted depth maps. Should be False. |

## 2. 3D object reconstruction with a depth sensor

| Parameter | Type | Description |
| :----- | :-----: | :----- |
| `numGPU` | int  | GPU device to use for evaluation. |
| `data_path` | str | Path to the data directory.  |
| `params_name` | str | Name of the config file corresponding to the surface coverage gain module to be evaluated.  |
| `scone_occ_model_name` | str | Name of the weights file corresponding to the occupancy probability module to be evaluated. |
| `scone_vis_model_name` | str | Name of the weights file corresponding to the surface coverage gain module to be evaluated. |
| `pc_size` | int | Number of points in the sequence processed by the self-attention unit of the surface coverage gain. |
| `n_view_max` | int |  Maximum number of views for reconstruction. Starting from one random initial view, the model iteratively predicts NBVs and captures up to `n_view_max` depth maps. |
| `test_novel` | bool | If True, starts a test on categories of objects never seen during training. |
| `results_json_name` | str | Name of the json file in which results will be saved. |