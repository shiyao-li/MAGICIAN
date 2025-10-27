import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import time

from .utils import *
from .spherical_harmonics import get_spherical_harmonics
from .CustomGeometry import *
from .CustomDataset import CustomShapenetDataset
from .spherical_harmonics import clear_spherical_harmonics_cache
from ..networks.SconeOcc import SconeOcc
from ..networks.SconeVis import SconeVis, KLDivCE, L1_loss, Uncentered_L1_loss

from .idr_torch import rank as idr_torch_rank
from .idr_torch import size as idr_torch_size
from .idr_torch import local_rank as idr_torch_local_rank
from .idr_torch import cpus_per_task as idr_torch_cpus_per_task
from .idr_torch import hostnames as idr_torch_hostnames
from .idr_torch import gpus_ids as idr_torch_gpus_ids


def setup_device(params, ddp_rank=None):

    if params.ddp:
        print("Setup device", str(ddp_rank), "for DDP training...")
        os.environ["CUDA_VISIBLE_DEVICES"] = params.CUDA_VISIBLE_DEVICES

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = str(params.WORLD_SIZE)
        os.environ['RANK'] = str(ddp_rank)

        dist.init_process_group("nccl", rank=ddp_rank, world_size=params.WORLD_SIZE)

        device = torch.device("cuda:" + str(ddp_rank))
        torch.cuda.set_device(device)

        torch.cuda.empty_cache()
        print("Setup done!")

        if ddp_rank == 0:
            print(torch.cuda.memory_summary())

    elif params.jz:
        print("Setup device", str(idr_torch_rank), " for Jean Zay training...")
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                rank=idr_torch_rank,
                                world_size=idr_torch_size)

        torch.cuda.set_device(idr_torch_local_rank)
        device = torch.device("cuda")

        torch.cuda.empty_cache()
        print("Setup done!")

        if idr_torch_rank == 0:
            print(torch.cuda.memory_summary())

    else:
        # Set our device:
        if torch.cuda.is_available():
            device = torch.device("cuda:" + str(params.numGPU))
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        print(device)

        # Empty cache
        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary())

    return device


def load_params(json_name, flatten=True):
    return Params(json_name, flatten=flatten)


def reduce_tensor(tensor: torch.Tensor, world_size):
    """Reduce tensor across all nodes."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def to_python_float(t: torch.Tensor):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def cleanup():
    dist.destroy_process_group()


def get_shapenet_dataloader(batch_size,
                            ddp=False, jz=False,
                            world_size=None, ddp_rank=None,
                            test_novel=False,
                            test_number=-1,
                            load_obj=False,
                            data_path=None,
                            shuffle=True):
    # Database path
    # SHAPENET_PATH = "../../../../datasets/shapenet2/ShapeNetCore.v2"
    memory_threshold = 10e6
    if data_path is None:
        if jz:
            SHAPENET_PATH = "../../datasets/ShapeNetCore.v1"
        else:
            # SHAPENET_PATH = "../../../../../../mnt/ssd/aguedon/ShapeNetCore.v1/ShapeNetCore.v1"
            SHAPENET_PATH = "../../../../datasets/ShapeNetCore.v1"
            # "../../datasets/ShapeNetCore.v1"
            # "../../../../../../mnt/ssd/aguedon/ShapeNetCore.v1/ShapeNetCore.v1"
    else:
        SHAPENET_PATH = data_path

    database_path = os.path.join(SHAPENET_PATH, "train_categories")
    train_json = os.path.join(SHAPENET_PATH, "train_list.json")
    val_json = os.path.join(SHAPENET_PATH, "val_list.json")
    if not test_novel:
        if test_number == 0:
            test_json = os.path.join(SHAPENET_PATH, "test_list.json")
        elif test_number == -1:
            test_json = os.path.join(SHAPENET_PATH, "all_test_list.json")
        else:
            test_json = os.path.join(SHAPENET_PATH, "test_list_" + str(test_number) + ".json")
            print("Using test split number " + str(test_number) + ".")
    else:
        database_path = os.path.join(SHAPENET_PATH, "test_categories")
        print("Using novel test split.")
        if test_number >= 0:
            test_json = os.path.join(SHAPENET_PATH, "test_novel_list.json")
        else:
            test_json = os.path.join(SHAPENET_PATH, "all_test_novel_list.json")
        # test_json = os.path.join(SHAPENET_PATH, "debug_list.json")
    # train_json = os.path.join(SHAPENET_PATH, "debug_train_list.json")

    train_dataset = CustomShapenetDataset(data_path=database_path,
                                          memory_threshold=memory_threshold,
                                          save_to_json=False,
                                          load_from_json=True,
                                          json_name=train_json,
                                          official_split=True,
                                          adjust_diagonally=True,
                                          load_obj=load_obj)
    val_dataset = CustomShapenetDataset(data_path=database_path,
                                        memory_threshold=memory_threshold,
                                        save_to_json=False,
                                        load_from_json=True,
                                        json_name=val_json,
                                        official_split=True,
                                        adjust_diagonally=True,
                                        load_obj=load_obj)
    test_dataset = CustomShapenetDataset(data_path=database_path,
                                         memory_threshold=memory_threshold,
                                         save_to_json=False,
                                         load_from_json=True,
                                         json_name=test_json,
                                         official_split=True,
                                         adjust_diagonally=True,
                                         load_obj=load_obj)

    if ddp or jz:
        if jz:
            rank = idr_torch_rank
        else:
            rank = ddp_rank
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=world_size,
                                           rank=rank,
                                           shuffle=shuffle,
                                           drop_last=True)
        valid_sampler = DistributedSampler(val_dataset,
                                           num_replicas=world_size,
                                           rank=rank,
                                           shuffle=False,
                                           drop_last=True)
        test_sampler = DistributedSampler(test_dataset,
                                          num_replicas=world_size,
                                          rank=rank,
                                          shuffle=False,
                                          drop_last=True)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      drop_last=True,
                                      collate_fn=collate_batched_meshes,
                                      sampler=train_sampler)
        validation_dataloader = DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           drop_last=True,
                                           collate_fn=collate_batched_meshes,
                                           sampler=valid_sampler)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     drop_last=True,
                                     collate_fn=collate_batched_meshes,
                                     sampler=test_sampler)
    else:
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      collate_fn=collate_batched_meshes,
                                      shuffle=shuffle)
        validation_dataloader = DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           collate_fn=collate_batched_meshes,
                                           shuffle=False)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     collate_fn=collate_batched_meshes,
                                     shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader


def get_optimizer(params, model):
    """
    Returns AdamW optimizer with linear warmup steps at beginning.

    :param params: (Params) Hyper parameters file.
    :param model: Model to be trained.
    :return: (Tuple) Optimizer and its name.
    """
    optimizer = WarmupConstantOpt(learning_rate=params.learning_rate,
                                  warmup=params.warmup,
                                  optimizer=torch.optim.AdamW(model.parameters(),
                                                              lr=0
                                                              )
                                  )
    opt_name = "WarmupAdamW"

    return optimizer, opt_name


def update_learning_rate(params, optimizer, learning_rate):

    if params.noam_opt:
        optimizer.model_size = 1 / (optimizer.warmup * learning_rate ** 2)
        if optimizer._step == 0:
            optimizer._rate = 0
        else:
            optimizer._rate = (optimizer.model_size ** (-0.5)
                               * min(optimizer._step ** (-0.5),
                                     optimizer._step * optimizer.warmup ** (-1.5)))

    else:
        optimizer.learning_rate = learning_rate
        if optimizer._step == 0:
            optimizer._rate = 0
        else:
            optimizer._rate = optimizer.learning_rate * min(1., optimizer._step / optimizer.warmup)


def initialize_scone_occ_weights(scone_occ, from_previous_model=None):
    # old init
    # scone_occ.apply(init_weights)

    # New init
    if from_previous_model is not None:
        previous_weights = {}
        for name, child in from_previous_model.named_modules():
            if isinstance(child, nn.Linear):
                with torch.no_grad():
                    layer = child.weight.cpu()
                    previous_weights[name] = {'mean': layer.mean().item(),
                                              'std': layer.std().item()}

    for name, child in scone_occ.named_modules():
        if isinstance(child, nn.Linear):
            layer_name = name.split('.')[-1]

            if from_previous_model is None:
                if layer_name in ['w_q', 'w_k', 'w_v']:
                    torch.nn.init.xavier_normal_(child.weight)
                    print(name, "initialized with Xavier normal.")
                else:
                    torch.nn.init.kaiming_normal_(child.weight, nonlinearity='relu')
                    print(name, "initialized with Kaiming normal.")
            else:
                layer_mean = previous_weights[name]['mean']
                layer_std = previous_weights[name]['std']
                torch.nn.init.normal_(child.weight, mean=layer_mean, std=layer_std)
                print(name, "initialized with normal (m, sigma) =", (layer_mean, layer_std))


def initialize_scone_occ(params, scone_occ, device,
                         torch_seed=None,
                         load_pretrained_weights=False,
                         pretrained_weights_name=None,
                         ddp_rank=None,
                         return_best_train_loss=False,
                         load_from_ddp_model=True):
    """
    Initializes SCONE's occupancy probability prediction module for training.
    Can be initialized from scratch, or from an already trained model to resume training.

    :param params: (Params) Hyper parameters file.
    :param scone_occ: (SconeOcc) Occupancy probability prediction model.
    :param device: Device.
    :param torch_seed: (int) Seed used to initialize the network.
    :param load_pretrained_weights: (bool) If True, pretrained weights are loaded for initialization.
    :param ddp_rank: Rank dor DDP training.
    :return: (Tuple) Initialized SconeOcc model, Optimizer, optimizer name, start epoch, best loss.
    If training from scratch, start_epoch=0 and best_loss=0.
    """
    model_name = params.scone_occ_model_name
    start_epoch = 0
    best_loss = 1000.
    best_train_loss = 1000.

    # Weight initialization process
    if load_pretrained_weights:
        # Load pretrained weights if needed
        if pretrained_weights_name==None:
            weights_file = "unvalidated_" + model_name + ".pth"
        else:
            weights_file = pretrained_weights_name
        weights_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../weights/scone/occupancy")
        weights_file = os.path.join(weights_dir, weights_file)
        checkpoint = torch.load(weights_file, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        best_train_loss = np.min(checkpoint['train_losses'])

        ddp_model = False
        if load_from_ddp_model is None:
            if (model_name[:2] == "jz") or (model_name[:3] == "ddp"):
                ddp_model = True
        else:
            ddp_model = load_from_ddp_model

        if ddp_model:
            scone_occ = load_ddp_state_dict(scone_occ, checkpoint['model_state_dict'])
        else:
            scone_occ.load_state_dict(checkpoint['model_state_dict'])

    else:
        # Else, applies a basic initialization process
        if torch_seed is not None:
            torch.manual_seed(torch_seed)
            print("Seed", torch_seed, "chosen.")
        initialize_scone_occ_weights(scone_occ)

    # DDP wrapping if needed
    if params.ddp:
        scone_occ = DDP(scone_occ,
                        device_ids=[ddp_rank])
    elif params.jz:
        scone_occ = DDP(scone_occ,
                        device_ids=[idr_torch_local_rank])

    # Creating optimizer
    optimizer, opt_name = get_optimizer(params, scone_occ)
    if load_pretrained_weights:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if return_best_train_loss:
        return scone_occ, optimizer, opt_name, start_epoch, best_loss, best_train_loss
    else:
        return scone_occ, optimizer, opt_name, start_epoch, best_loss


def load_scone_occ(params, trained_model_name, ddp_model, device):
    """
    Loads an already trained occupancy probability prediction model for inference on a single GPU.

    :param params: (Params) Parameters file.
    :param trained_model_name: (str) Name of trained model's checkpoint.
    :param device: Device.
    :return: (SconeOcc) Occupancy probability prediction module with trained weights.
    """
    scone_occ = SconeOcc().to(device)

    model_name = params.scone_occ_model_name

    # Loads checkpoint
    weights_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../weights/scone/occupancy")
    weights_file = os.path.join(weights_dir, trained_model_name)
    checkpoint = torch.load(weights_file, map_location=device)
    print("Model name:", trained_model_name)
    print("Trained for", checkpoint['epoch'], 'epochs.')
    print("Training finished with loss", checkpoint['loss'])

    # Loads trained weights
    if ddp_model:
        scone_occ = load_ddp_state_dict(scone_occ, checkpoint['model_state_dict'])
    else:
        scone_occ.load_state_dict(checkpoint['model_state_dict'])

    return scone_occ


def initialize_scone_vis_weights(scone_vis, from_previous_model=None):
    # old init
    # scone_vis.apply(init_weights)

    # New init
    if from_previous_model is not None:
        previous_weights = {}
        for name, child in from_previous_model.named_modules():
            if isinstance(child, nn.Linear):
                with torch.no_grad():
                    layer = child.weight.cpu()
                    previous_weights[name] = {'mean': layer.mean().item(),
                                              'std': layer.std().item()}

    for name, child in scone_vis.named_modules():
        if isinstance(child, nn.Linear):
            layer_name = name.split('.')[-1]

            if from_previous_model is None:
                if layer_name in ['w_q', 'w_k', 'w_v']:
                    torch.nn.init.xavier_normal_(child.weight)
                    print(name, "initialized with Xavier normal.")
                else:
                    torch.nn.init.kaiming_normal_(child.weight, nonlinearity='relu')
                    print(name, "initialized with Kaiming normal.")
            else:
                layer_mean = previous_weights[name]['mean']
                layer_std = previous_weights[name]['std']
                torch.nn.init.normal_(child.weight, mean=layer_mean, std=layer_std)
                print(name, "initialized with normal (m, sigma) =", (layer_mean, layer_std))


def initialize_scone_vis(params, scone_vis, device,
                         torch_seed=None,
                         load_pretrained_weights=False,
                         pretrained_weights_name=None,
                         ddp_rank=None,
                         return_best_train_loss=False,
                         load_from_ddp_model=True):
    """
    Initializes SCONE's visibility prediction module for training.
    Can be initialized from scratch, or from an already trained model to resume training.

    :param params: (Params) Hyper parameters file.
    :param scone_vis: (SconeVis) Visibility prediction model.
    :param device: Device.
    :param torch_seed: (int) Seed used to initialize the network.
    :param load_pretrained_weights: (bool) If True, pretrained weights are loaded for initialization.
    :param ddp_rank: Rank dor DDP training.
    :return: (Tuple) Initialized SconeVis model, Optimizer, optimizer name, start epoch, best loss, best coverage.
    If training from scratch, start_epoch=0, best_loss=0. and best_coverage=0.
    """
    model_name = params.scone_vis_model_name
    start_epoch = 0
    best_loss = 1000.
    best_coverage = 0.
    best_train_loss = 1000.

    # Weight initialization process
    if load_pretrained_weights:
        # Load pretrained weights if needed
        if pretrained_weights_name==None:
            weights_file = "unvalidated_" + model_name + ".pth"
        else:
            weights_file = pretrained_weights_name
        weights_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../weights/scone/coverage_gain")
        weights_file = os.path.join(weights_dir, weights_file)
        checkpoint = torch.load(weights_file, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        best_train_loss = np.min(checkpoint['train_losses'])
        if 'coverage' in checkpoint:
            best_coverage = checkpoint['coverage']

        ddp_model = False
        if load_from_ddp_model is None:
            if (model_name[:2] == "jz") or (model_name[:3] == "ddp"):
                ddp_model = True
        else:
            ddp_model = load_from_ddp_model

        if ddp_model:
            scone_vis = load_ddp_state_dict(scone_vis, checkpoint['model_state_dict'])
        else:
            scone_vis.load_state_dict(checkpoint['model_state_dict'])

    else:
        # Else, applies a basic initialization process
        if torch_seed is not None:
            torch.manual_seed(torch_seed)
            print("Seed", torch_seed, "chosen.")
        initialize_scone_vis_weights(scone_vis)

    # DDP wrapping if needed
    if params.ddp:
        scone_vis = DDP(scone_vis,
                        device_ids=[ddp_rank])
    elif params.jz:
        scone_vis = DDP(scone_vis,
                        device_ids=[idr_torch_local_rank])

    # Creating optimizer
    optimizer, opt_name = get_optimizer(params, scone_vis)
    if load_pretrained_weights:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if return_best_train_loss:
        return scone_vis, optimizer, opt_name, start_epoch, best_loss, best_coverage, best_train_loss
    else:
        return scone_vis, optimizer, opt_name, start_epoch, best_loss, best_coverage


def load_scone_vis(params, trained_model_name, ddp_model, device):
    """
    Loads an already trained visibility prediction model for inference on a single GPU.

    :param params: (Params) Parameters file.
    :param trained_model_name: (str) Name of trained model's checkpoint.
    :param device: Device.
    :return: (SconeVis) Visibility prediction module with trained weights.
    """
    scone_vis = SconeVis(use_sigmoid=params.use_sigmoid).to(device)

    model_name = params.scone_vis_model_name

    # Loads checkpoint
    weights_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../weights/scone/coverage_gain")
    weights_file = os.path.join(weights_dir, trained_model_name)
    checkpoint = torch.load(weights_file, map_location=device)
    print("Model name:", trained_model_name)
    print("Trained for", checkpoint['epoch'], 'epochs.')
    print("Training finished with loss", checkpoint['loss'])

    # Loads trained weights
    if ddp_model:
        scone_vis = load_ddp_state_dict(scone_vis, checkpoint['model_state_dict'])
    else:
        scone_vis.load_state_dict(checkpoint['model_state_dict'])

    return scone_vis


def get_cov_loss_fn(params):
    if params.training_loss == "kl_divergence":
        cov_loss_fn = KLDivCE()

    elif params.training_loss == "l1":
        cov_loss_fn = L1_loss()

    elif params.training_loss == "uncentered_l1":
        cov_loss_fn = Uncentered_L1_loss()

    else:
        raise NameError("Invalid training loss function."
                        "Please choose a valid loss between 'kl_divergence', 'l1' or 'uncentered_l1.")

    return cov_loss_fn


def get_occ_loss_fn(params):
    if params.training_loss == "mse":
        occ_loss_fn = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        return occ_loss_fn

    else:
        raise NameError("Invalid training loss function."
                        "Please choose a valid loss like 'mse'.")


# -----Data Functions-----

# TO CHANGE! Maybe Load on CPU for large scenes.
def get_gt_partial_point_clouds(path, device, normalization_factor=None):
    """
    Loads ground truth partial point clouds for training.
    :param path:
    :param device:
    :param normalization_factor: factor to normalize the point cloud.
    if None, the point cloud is not normalized.
    :return:
    """
    parent_dir = os.path.dirname(path)
    load_directory = os.path.join(parent_dir, "tensors")

    file_name = "partial_point_clouds.pt"
    pc_dict = torch.load(os.path.join(load_directory, file_name),
                         map_location=device)

    part_pc = pc_dict['partial_point_cloud']
    coverage = torch.vstack(pc_dict['coverage'])

    if (normalization_factor is not None) and (normalization_factor != 1.):
        for i in range(len(part_pc)):
            part_pc[i] = normalization_factor * part_pc[i]

    return part_pc, coverage


def get_gt_occupancy_field(path, device):
    """
    Loads ground truth occupancy field for training.
    :param path:
    :param device:
    :return:
    """
    parent_dir = os.path.dirname(path)
    load_directory = os.path.join(parent_dir, "tensors")

    file_name = "occupancy_field.pt"

    pc_dict = torch.load(os.path.join(load_directory, file_name),
                         map_location=device)

    X_world = pc_dict['occupancy_field'][..., :3]
    occs = pc_dict['occupancy_field'][..., 3:]

    return X_world, occs


def get_gt_surface(params, path, device, normalization_factor=None):
    parent_dir = os.path.dirname(path)
    load_directory = os.path.join(parent_dir, "tensors")
    file_name = "surface_points.pt"

    surface_dict = torch.load(os.path.join(load_directory, file_name),
                              map_location=device)
    gt_surface = surface_dict['surface_points']

    if params.surface_epsilon_is_constant:
        surface_epsilon = params.surface_epsilon
    else:
        surface_epsilon = surface_dict['epsilon']

    if (normalization_factor is not None) and (normalization_factor != 1.):
        gt_surface = gt_surface * normalization_factor
        surface_epsilon = surface_epsilon * normalization_factor

    return gt_surface, surface_epsilon


def get_optimal_sequence(optimal_sequences, mesh_path, n_views):
    key = os.path.basename(os.path.dirname(mesh_path))

    # optimal_seq = optimal_sequences[key]['idx']
    optimal_seq = torch.Tensor(optimal_sequences[key]['idx']).long()
    seq_coverage = optimal_sequences[key]['coverage']

    return optimal_seq[:n_views], seq_coverage[:n_views]


def compute_gt_coverage_gain_from_precomputed_matrices(coverage, initial_cam_idx):
    device = coverage.device

    n_camera_candidates, n_points_surface = coverage.shape[0], coverage.shape[1]

    # Compute coverage matrix of previous cameras, and the corresponding value
    coverage_matrix = torch.sum(coverage[initial_cam_idx], dim=0).view(1, n_points_surface).expand(n_camera_candidates, -1)
    previous_coverage = torch.mean(torch.heaviside(coverage_matrix,
                                                   values=torch.zeros_like(coverage_matrix,
                                                                           device=device)), dim=-1)
    # Compute coverage matrices of previous + new camera for every camera
    coverage_matrix = coverage_matrix + coverage
    coverage_matrix = torch.mean(torch.heaviside(coverage_matrix,
                                                 values=torch.zeros_like(coverage_matrix,
                                                                         device=device)), dim=-1)

    # Compute coverage gain value
    coverage_matrix = coverage_matrix - previous_coverage

    return coverage_matrix.view(-1, 1)


def compute_surface_coverage_from_cam_idx(coverage, cam_idx):
    device = coverage.device

    coverage_matrix = torch.sum(coverage[cam_idx], dim=0)

    coverage = torch.mean(torch.heaviside(coverage_matrix,
                                          values=torch.zeros_like(coverage_matrix,
                                                                  device=device)), dim=-1)

    return coverage.view(1)


def get_validation_n_views_list(params, dataloader):
    n_views = params.n_view_max - params.n_view_min + 1

    n_views_list = np.repeat(np.arange(start=params.n_view_min,
                                       stop=params.n_view_max + 1).reshape(1, n_views),
                             len(dataloader.dataset) // n_views + 1, axis=0).reshape(-1)

    return n_views_list


def get_validation_n_view(params, n_views_list, batch, rank):
    idx = batch * params.total_batch_size + rank * params.batch_size

    return n_views_list[idx:idx + params.batch_size]


def get_validation_optimal_sequences(jz, device):
    # if jz:
    #     SHAPENET_PATH = "../../datasets/ShapeNetCore.v1"
    # else:
    #     SHAPENET_PATH = "../../../../../../mnt/ssd/aguedon/ShapeNetCore.v1/ShapeNetCore.v1"

    SHAPENET_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../data/ShapeNetCore.v1")

    file_name = "validation_optimal_trajectories.pt"
    optimal_sequences = torch.load(os.path.join(SHAPENET_PATH, file_name),
                                   map_location=device)

    return optimal_sequences


def get_all_harmonics_under_degree(degree, n_elev, n_azim, device):
    """
    Gets values for all harmonics with l < degree.
    :param degree:
    :param n_elev:
    :param n_azim:
    :param device:
    :return:
    """
    h_elev = torch.Tensor(
        [-np.pi / 2 + (i + 1) / (n_elev + 1) * np.pi for i in range(n_elev) for j in range(n_azim)]).to(device)
    h_polar = -h_elev + np.pi / 2

    h_azim = torch.Tensor([2 * np.pi * j / n_azim for i in range(n_elev) for j in range(n_azim)]).to(device)

    z = torch.zeros([i for i in h_polar.shape] + [0], device=h_polar.device)

    clear_spherical_harmonics_cache()
    for l in range(degree):
        y = get_spherical_harmonics(l, h_polar, h_azim)
        z = torch.cat((z, y), dim=-1)

    z = z.transpose(dim0=0, dim1=1)

    return z, h_polar, h_azim


def get_cameras_on_sphere(params, device, pole_cameras=False, n_elev=None, n_azim=None, camera_dist=None):
    """
    Returns cameras candidate positions, sampled on a sphere.
    Made for SCONE pretraining on ShapeNet.
    :param params: (Params) The dictionary of parameters.
    :param device:
    :return: A tuple of Tensors (X_cam, candidate_dist, candidate_elev, candidate_azim)
    X_cam has shape (n_camera_candidate, 3)
    All other tensors have shape (n_camera candidate, )
    """
    if n_elev is None or n_azim is None:
        n_elev = params.n_camera_elev
        n_azim = params.n_camera_azim
        n_camera = params.n_camera
    else:
        n_camera = n_elev * n_azim
        if pole_cameras:
            n_camera += 2

    if camera_dist is None:
        camera_dist = params.camera_dist

    candidate_dist = torch.Tensor([camera_dist for i in range(n_camera)]).to(device)

    candidate_elev = [-90. + (i + 1) / (n_elev + 1) * 180.
                      for i in range(n_elev)
                      for j in range(n_azim)]

    candidate_azim = [360. * j / n_azim
                      for i in range(n_elev)
                      for j in range(n_azim)]

    if pole_cameras:
        candidate_elev = [-89.9] + candidate_elev + [89.9]
        candidate_azim = [0.] + candidate_azim + [0.]

    candidate_elev = torch.Tensor(candidate_elev).to(device)
    candidate_azim = torch.Tensor(candidate_azim).to(device)

    X_cam = get_cartesian_coords(r=candidate_dist.view(-1, 1),
                                 elev=candidate_elev.view(-1, 1),
                                 azim=candidate_azim.view(-1, 1),
                                 in_degrees=True)

    return X_cam, candidate_dist, candidate_elev, candidate_azim


def normalize_points_in_prediction_box(points, prediction_box_center, prediction_box_diag):
    """

    :param points:
    :param prediction_box_center:
    :param prediction_box_diag:
    :return:
    """
    return (points - prediction_box_center) / prediction_box_diag


def compute_view_state(pts, X_view, n_elev, n_azim):
    """
    Computes view_state vector for points pts and camera positions X_view.
    :param pts: Tensor with shape (n_cloud, seq_len, pts_dim) where pts_dim >= 3.
    :param X_view: Tensor with shape (n_screen_cameras, 3).
    Represents camera positions in prediction camera space coordinates.
    :param n_elev: Integer. Number of elevations values to discretize view states.
    :param n_azim: Integer. Number of azimuth values to discretize view states
    :return: A Tensor with shape (n_cloud, seq_len, n_elev*n_azim).
    """
    # Initializing variables
    device = pts.device
    n_view = len(X_view)
    n_clouds, seq_len, _ = pts.shape
    n_candidates = n_elev * n_azim

    elev_step = np.pi / (n_elev + 1)
    azim_step = 2 * np.pi / n_azim

    X_pts = pts[..., :3]

    # Computing camera elev and azim in every pts space coordinates
    rays = X_view.view(1, 1, n_view, 3).expand(n_clouds, seq_len, -1, -1) \
           - X_pts.view(n_clouds, seq_len, 1, 3).expand(-1, -1, n_view, -1)

    _, ray_elev, ray_azim = get_spherical_coords(rays.view(-1, 3))

    ray_elev = ray_elev.view(n_clouds, seq_len, n_view)
    ray_azim = ray_azim.view(n_clouds, seq_len, n_view)

    # Projecting elev and azim to the closest values in the discretized cameras
    idx_elev = floor_divide(ray_elev, elev_step)
    idx_azim = floor_divide(ray_azim, azim_step)

    # If closest to ceil than floor, we add 1
    idx_elev[ray_elev % elev_step > elev_step / 2.] += 1
    idx_azim[ray_azim % azim_step > azim_step / 2.] += 1

    # Elevation can't be below minimal or above maximal values
    idx_elev[idx_elev >= n_elev] = n_elev - 1
    idx_elev[idx_elev < -n_elev // 2] = -n_elev // 2

    # If azimuth is greater than 180 degrees, we reset it back to -180 degrees
    idx_azim[idx_azim > n_azim // 2] = -n_azim // 2

    # Normalizing indices to retrieve camera positions in flattened view_state
    idx_elev += n_elev // 2
    idx_azim[idx_azim < 0] += n_azim

    indices = idx_elev.long() * n_azim + idx_azim.long()
    indices %= n_candidates
    q = torch.arange(start=0, end=n_clouds * seq_len, step=1, device=device).view(-1, 1).expand(-1, n_view)

    flat_indices = indices.view(-1, n_view)
    flat_indices = q * n_candidates + flat_indices
    flat_indices = flat_indices.view(-1)

    # Compute view_state and set visited camera values to 1
    view_state = torch.zeros(n_clouds, seq_len, n_candidates, device=device)
    view_state.view(-1)[flat_indices] = 1.

    return view_state


def move_view_state_to_view_space(view_state, fov_camera, n_elev, n_azim):
    """
    "Rotate" the view state vectors to the corresponding view space.

    :param view_state: (Tensor) View state tensor with shape (n_cloud, seq_len, n_elev * n_azim)
    :param fov_camera: (FoVPerspectiveCamera)
    :param n_elev: (int)
    :param n_azim: (int)
    :return: Rotated view state tensor with shape (n_cloud, seq_len, n_elev * n_azim)
    """
    device = view_state.device
    n_clouds = view_state.shape[0]
    seq_len = view_state.shape[1]

    n_view = n_elev * n_azim

    candidate_dist = torch.Tensor([1. for i in range(n_elev * n_azim)]).to(device)

    candidate_elev = [-90. + (i + 1) / (n_elev + 1) * 180.
                      for i in range(n_elev)
                      for j in range(n_azim)]

    candidate_azim = [360. * j / n_azim
                      for i in range(n_elev)
                      for j in range(n_azim)]

    candidate_elev = torch.Tensor(candidate_elev).to(device)
    candidate_azim = torch.Tensor(candidate_azim).to(device)

    X_cam_ref = get_cartesian_coords(r=candidate_dist.view(-1, 1),
                                     elev=candidate_elev.view(-1, 1),
                                     azim=candidate_azim.view(-1, 1),
                                     in_degrees=True)
    X_cam_inv = fov_camera.get_world_to_view_transform().inverse().transform_points(
        X_cam_ref) - fov_camera.get_camera_center()

    elev_step = np.pi / (n_elev + 1)
    azim_step = 2 * np.pi / n_azim

    _, ray_elev, ray_azim = get_spherical_coords(X_cam_inv.view(-1, 3))

    ray_elev = ray_elev.view(n_view)
    ray_azim = ray_azim.view(n_view)

    # Projecting elev and azim to the closest values in the discretized cameras
    idx_elev = floor_divide(ray_elev, elev_step)
    idx_azim = floor_divide(ray_azim, azim_step)

    # If closest to ceil than floor, we add 1
    idx_elev[ray_elev % elev_step > elev_step / 2.] += 1
    idx_azim[ray_azim % azim_step > azim_step / 2.] += 1

    # Elevation can't be below minimal or above maximal values
    idx_elev[idx_elev > n_elev // 2] = n_elev // 2
    idx_elev[idx_elev < -(n_elev // 2)] = -(n_elev // 2)

    # If azimuth is greater than 180 degrees, we reset it back to -180 degrees
    idx_azim[idx_azim > n_azim // 2] = -(n_azim // 2)

    # Normalizing indices to retrieve camera positions in flattened view_state
    idx_elev += n_elev // 2
    idx_azim[idx_azim < 0] += n_azim

    indices = idx_elev.long() * n_azim + idx_azim.long()

    rot_view_state = torch.gather(input=view_state, dim=2, index=indices.view(1, 1, -1).expand(n_clouds, seq_len, -1))

    return rot_view_state



def compute_view_harmonics(view_state, base_harmonics, h_polar, h_azim, n_elev, n_azim):
    """
    Computes spherical harmonics corresponding to the view_state vector.
    :param view_state: Tensor with shape (n_cloud, seq_len, n_elev*n_azim).
    :param base_harmonics: Tensor with shape (n_harmonics, n_elev*n_azim).
    :param h_polar:
    :param h_azim:
    :param n_elev:
    :param n_azim:
    :return: Tensor with shape (n_cloud, seq_len, n_harmonics)
    """
    # Define parameters
    n_harmonics = base_harmonics.shape[0]
    n_clouds, seq_len, n_values = view_state.shape

    polar_step = np.pi / (n_elev + 1)
    azim_step = 2 * np.pi / n_azim

    # Expanding variables to parallelize computation
    all_values = view_state.view(n_clouds, seq_len, 1, n_values).expand(-1, -1, n_harmonics, -1)
    all_polar = h_polar.view(1, 1, 1, n_values).expand(n_clouds, seq_len, n_harmonics, -1)
    # all_harmonics = base_harmonics.view(1, 1, n_harmonics, n_values).expand(n_clouds, seq_len, -1, -1)

    # Computing spherical L2-dot product on last axis
    coordinates = torch.sum(all_values * base_harmonics * torch.sin(all_polar) * polar_step * azim_step, dim=-1)

    return coordinates


# ----------Model functions----------

def compute_occupancy_probability(scone_occ, pc, X, view_harmonics, mask=None,
                                  max_points_per_pass=20000):
    """

    :param scone_occ: (Scone_Occ) SCONE's Occupancy Probability prediction model.
    :param pc: (Tensor) Input point cloud tensor with shape (n_clouds, seq_len, pts_dim)
    :param X: (Tensor) Input query points tensor with shape (n_clouds, n_sample, x_dim)
    :param view_harmonics: (Tensor) View state harmonic features. Tensor with shape (n_clouds, seq_len, n_harmonics)
    :param max_points_per_pass: (int) Maximal number of points per forward pass.
    :return:
    """
    n_clouds, seq_len, pts_dim = pc.shape[0], pc.shape[1], pc.shape[2]
    n_sample, x_dim = X.shape[1], X.shape[2]
    n_harmonics = view_harmonics.shape[2]

    preds = torch.zeros(n_clouds, 0, 1).to(X.device)

    p = max_points_per_pass // n_clouds
    q = n_sample // p
    r = n_sample % p
    n_loop = q
    if r != 0:
        n_loop += 1

    for i in range(n_loop):
        low_idx = i * p
        up_idx = (i + 1) * p
        if i == q:
            up_idx = q * p + r
        preds_i = scone_occ(pc, X[:, low_idx:up_idx], view_harmonics[:, low_idx:up_idx], verbose=False) # todo: to remove
        preds_i = preds_i.view(n_clouds, up_idx - low_idx, -1)
        preds = torch.cat((preds, preds_i), dim=1)

    return preds


def filter_proxy_points(view_cameras, X, pc, filter_tol=0.01):
    """
    Filter proxy points considering camera field of view and partial surface point cloud.
    WARNING: Works for a single scene! So X must have shape (n_proxy_points, 3)!
    :param view_cameras:
    :param X: (Tensor) Proxy points tensor with shape ()
    :param pc: (Tensor)
    :param filter_tol:
    :return:
    """

    n_view = view_cameras.R.shape[0]

    if (len(X.shape) != 2) or (len(pc.shape) != 2):
        raise NameError("Wrong shapes! X must have shape (n_proxy_points, 3) and pc must have shape (N, 3).")

    view_projection_transform = view_cameras.get_full_projection_transform()
    X_proj = view_projection_transform.transform_points(X)[..., :2].view(n_view, -1, 2)
    pc_proj = view_projection_transform.transform_points(pc)[..., :2].view(n_view, -1, 2)

    max_proj = torch.max(pc_proj, dim=-2, keepdim=True)[0].expand(-1, X_proj.shape[-2], -1)
    min_proj = torch.min(pc_proj, dim=-2, keepdim=True)[0].expand(-1, X_proj.shape[-2], -1)

    filter_mask = torch.prod((X_proj < max_proj + filter_tol) * (X_proj > min_proj - filter_tol), dim=0)
    filter_mask = torch.prod(filter_mask, dim=-1).bool()

    return X[filter_mask], filter_mask


def sample_proxy_points(X_world, preds, view_harmonics, n_sample, min_occ, use_occ_to_sample=True,
                        return_index=False):
    """

    :param X: Tensor with shape (n_points, 3)
    :param preds: Tensor with shape (n_points, 1)
    :param view_harmonics: Tensor with shape (n_points, n_harmonics)
    :param n_sample: integer
    :return:
    """
    mask = preds[..., 0] > min_occ
    res_X = X_world[mask]
    res_preds = preds[mask]
    res_harmonics = view_harmonics[mask]

    device = res_X.device
    n_points = res_X.shape[0]

    if use_occ_to_sample:
        sample_probs = res_preds[..., 0] / torch.sum(res_preds)
        sample_probs = torch.cumsum(sample_probs, dim=-1)

        samples = torch.rand(n_sample, 1, device=device)

        res_idx = sample_probs.view(1, n_points).expand(n_sample, -1) - samples.expand(-1, n_points)
        res_idx[res_idx < 0] = 2
        res_idx = torch.argmin(res_idx, dim=-1)

        res_idx, inverse_idx = torch.unique(res_idx, dim=0, return_inverse=True)

        res = torch.cat((res_X[res_idx], res_preds[res_idx]), dim=-1)
        res_harmonics = res_harmonics[res_idx]
        # res = torch.unique(res, dim=0)

    else:
        if len(res_X) > n_sample:
            res_X = res_X[:n_sample]
            res_preds = res_preds[:n_sample]
            res_harmonics = res_harmonics[:n_sample]

        res = torch.cat((res_X, res_preds), dim=-1)
        inverse_idx = None

    if return_index:
        return res, res_harmonics, inverse_idx
    else:
        return res, res_harmonics
