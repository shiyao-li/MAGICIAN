import argparse
from macarons.trainers.pretrain_scone_occ import *

dir_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(dir_path, "./data/ShapeNetCore.v1")
weights_dir = os.path.join(dir_path, "./weights/scone/occupancy")
configs_dir = os.path.join(dir_path, "./configs/scone/occupancy")

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to pretrain occupancy probability module using '
                                                 'ShapeNetCore.v1 meshes.')
    parser.add_argument('-c', '--config', type=str, help='name of the config file. '
                                                         'Default is "occupancy_pretraining_config.json".')

    args = parser.parse_args()

    if args.config:
        json_name = args.config
    else:
        json_name = "occupancy_pretraining_config.json.json"

    print("Using json name given in argument:")
    print(json_name)

    json_path = os.path.join(configs_dir, json_name)
    params = load_params(json_path)

    if params.ddp:
        mp.spawn(run_training,
                 args=(params,
                       ),
                 nprocs=params.WORLD_SIZE
                 )

    elif params.jz:
        run_training(params=params)

    else:
        run_training(params=params)
