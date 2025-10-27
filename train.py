import argparse
from macarons.trainers.train_macarons import *

dir_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(dir_path, "./data/scenes")
weights_dir = os.path.join(dir_path, "./weights/macarons")
configs_dir = os.path.join(dir_path, "./configs/macarons")

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to train a full macarons model in large 3D scenes.')
    parser.add_argument('-c', '--config', type=str, help='name of the config file. '
                                                         'Default is "macarons_default_training_config.json".')

    args = parser.parse_args()

    if args.config:
        json_name = args.config
    else:
        json_name = "macarons_default_training_config.json"

    print("Using json name given in argument:")
    print(json_name)

    json_path = os.path.join(configs_dir, json_name)
    params = load_params(json_path)
    # torch.autograd.set_detect_anomaly(params.anomaly_detection)

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
