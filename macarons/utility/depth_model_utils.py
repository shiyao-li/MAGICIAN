import numpy as np
import json
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import gc
from collections import OrderedDict
from pytorch3d.transforms import (axis_angle_to_matrix,
                                  euler_angles_to_matrix,
                                  matrix_to_euler_angles,
                                  matrix_to_quaternion,
                                  quaternion_apply)
from torchvision.transforms.functional import (adjust_brightness,
                                               adjust_contrast,
                                               adjust_saturation,
                                               adjust_hue,
                                               hflip,
                                               pad)

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
    quaternion_to_axis_angle,
)

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import time

from .utils import (
    init_weights,
    load_ddp_state_dict,
    count_parameters,
    Params,
    NoamOpt,
    WarmupConstantOpt)
from .CustomDataset import RGBDataset
from ..networks.ManyDepth import (
    FeatureExtractor,
    ExpansionLayer,
    DisparityLayer,
    CostVolumeBuilder,
    DepthDecoder,
    PoseDecoder,
    ManyDepth,
    SSIM,
    create_many_depth_model,
)
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


def load_params(json_name, flatten=False):
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


def transpose_channels(img, channel_is_at_the_end):
    if channel_is_at_the_end:
        res = 0. + torch.transpose(img, dim0=-1, dim1=-2)
        res = torch.transpose(res, dim0=-2, dim1=-3)
    else:
        res = 0. + torch.transpose(img, dim0=-3, dim1=-2)
        res = torch.transpose(res, dim0=-2, dim1=-1)
    return res


def get_dataloader(train_scenes, val_scenes, test_scenes,
                   batch_size,
                   test_ratio=1.,
                   ddp=False, jz=False,
                   world_size=None, ddp_rank=None,
                   alpha_max=1, use_future_images=False,
                   data_path=None):
    # Database path
    if data_path is None:
        if jz:
            database_path = "../../datasets/scenes/rgb"
        else:
            database_path = "../../../../datasets/scenes/rgb"
    else:
        database_path = data_path

    # Dataset
    train_dataset = RGBDataset(data_path=database_path,
                               alpha_max=alpha_max,
                               use_future_images=use_future_images,
                               scene_names=train_scenes)

    val_dataset = RGBDataset(data_path=database_path,
                               alpha_max=alpha_max,
                               use_future_images=use_future_images,
                               scene_names=val_scenes)

    test_dataset = RGBDataset(data_path=database_path,
                               alpha_max=alpha_max,
                               use_future_images=use_future_images,
                               scene_names=test_scenes)

    if ddp or jz:
        if jz:
            rank = idr_torch_rank
        else:
            rank = ddp_rank
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=world_size,
                                           rank=rank,
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
                                      shuffle=True)
        validation_dataloader = DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           collate_fn=collate_batched_meshes,
                                           shuffle=False)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     collate_fn=collate_batched_meshes,
                                     shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader


def load_pretrained_depth_model(pretrained_model_path, device):
    # ---Method 1
    model = torch.load(pretrained_model_path, map_location=device).to(device)

    # ---Method 2
    # model = create_many_depth_model(device=device, pretrained_resnet_path="pretrained_resnet18.pth")
    print("Pretrained depth model loaded. Using ResNet18 features.")
    return model.float()


def get_optimizer(params, model):
    if params.noam_opt:
        optimizer = NoamOpt(params.warmup_rate, params.warmup,
                            torch.optim.Adam(model.parameters(),
                                             lr=0,
                                             betas=(0.9, 0.98),
                                             eps=1e-9  # 1e-9
                                             )
                            )
        opt_name = "Noam"

    else:
        optimizer = WarmupConstantOpt(learning_rate=params.learning_rate,
                                      warmup=params.warmup,
                                      optimizer=torch.optim.AdamW(model.parameters(),
                                                                  lr=0
                                                                  )
                                      )
        # optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
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


def initialize_depth_model(params, depth_model, device,
                           torch_seed=None, initialize=True,
                           pretrained=True, ddp_rank=None):
    """
    Initializes an information model for training.
    Can be initialized from scratch, or from an already trained model to resume training.
    :param params:
    :param info_model:
    :param device:
    :param torch_seed:
    :param initialize:
    :param ddp_rank:
    :return:
    """
    model_name = params.depth_model_name
    start_epoch = 0
    best_loss = 1000.

    # Weight initialization if needed
    if initialize:
        if torch_seed is not None:
            torch.manual_seed(torch_seed)
            print("Seed", torch_seed, "chosen.")

            if not pretrained:
                depth_model.apply(init_weights)

    # Load previous training weights if needed
    if not initialize:
        checkpoint = torch.load("unvalidated_" + model_name + ".pth", map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']

        ddp_model = False
        if (model_name[:2] == "jz") or (model_name[:3] == "ddp"):
            ddp_model = True

        if ddp_model:
            depth_model = load_ddp_state_dict(depth_model, checkpoint['model_state_dict'])
        else:
            depth_model.load_state_dict(checkpoint['model_state_dict'])

    # DDP wrapping if needed
    if params.ddp:
        depth_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(depth_model)
        depth_model = DDP(depth_model,
                          device_ids=[ddp_rank])
    elif params.jz:
        depth_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(depth_model)
        depth_model = DDP(depth_model,
                          device_ids=[idr_torch_local_rank])

    # Creating optimizer
    optimizer, opt_name = get_optimizer(params, depth_model)
    if not initialize:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return depth_model, optimizer, opt_name, start_epoch, best_loss


def load_depth_model(params, trained_model_name, device):
    """

    :param params:
    :param trained_model_name:
    :param device:
    :return:
    """
    depth_model = load_pretrained_depth_model(pretrained_model_path=params.pretrained_model_path,
                                              device=device)

    model_name = params.depth_model_name

    # Load trained weights
    checkpoint = torch.load(trained_model_name, map_location=device)
    print("Model name:", trained_model_name)
    print("Trained for", checkpoint['epoch'], 'epochs.')
    print("Training finished with loss", checkpoint['loss'])

    ddp_model = False
    if (model_name[:2] == "jz") or (model_name[:3] == "ddp"):
        ddp_model = True

    if ddp_model:
        depth_model = load_ddp_state_dict(depth_model, checkpoint['model_state_dict'])
    else:
        depth_model.load_state_dict(checkpoint['model_state_dict'])

    return depth_model


def get_relative_pose_matrices(R, alpha_R, T, alpha_T):
    batch_size, n_alpha = T.shape[0], alpha_T.shape[1]

    expanded_R = R.view(batch_size, 1, 3, 3).expand(-1, n_alpha, -1, -1)
    expanded_T = T.view(batch_size, 1, 3).expand(-1, n_alpha, -1)

    relative_R = expanded_R.transpose(dim0=-1, dim1=-2) @ alpha_R
    relative_T = alpha_T - quaternion_apply(matrix_to_quaternion(alpha_R.transpose(dim0=-1, dim1=-2) @ expanded_R),
                                            expanded_T)

    return relative_R, relative_T


def get_pose_loss_fn(params):
    """
    Returns the following function.
    :param params:
    :return:
    """
    def pose_loss(pred_pose, truth_pose, pose_factor, rotation_mode, epsilon=1e-9):
        """

        :param pred_pose:
        :param truth_pose:
        :param pose_factor:
        :param rotation_mode: String. Can be 'angle' or 'matrix'.
        :param epsilon:
        :return:
        """
        pose_size = pred_pose.shape[-1]
        batch_size = pred_pose.shape[0]

        pred_relative_R = pose_factor * pred_pose[..., 3:]
        pred_relative_T = pose_factor * pred_pose[..., :3].view(-1, 3)

        truth_relative_R = pose_factor * truth_pose[..., 3:]
        truth_relative_T = pose_factor * truth_pose[..., :3].view(-1, 3)

        if rotation_mode == 'matrix':
            pred_relative_R = (axis_angle_to_matrix(pred_relative_R)).view(-1, 3*3)
            truth_relative_R = (axis_angle_to_matrix(truth_relative_R)).view(-1, 3 * 3)
        if rotation_mode == 'angle':
            pred_relative_R /= np.pi
            truth_relative_R /= np.pi
        else:
            raise NameError("Rotation mode is invalid. "
                            "Please select one of the following rotation modes: 'matrix', 'angle'.")

        M = 1.  # 0.1

        return M * torch.nn.functional.mse_loss(pred_relative_R,
                                                truth_relative_R,
                                                reduction='mean'
                                                ) + M * torch.nn.functional.mse_loss(pred_relative_T,
                                                                                     truth_relative_T,
                                                                                     reduction='mean'
                                                                                     )

    return pose_loss


def get_depth_loss_fn(params):
    """
    Returns the following function.
    :param params:
    :return:
    """
    def depth_loss(pred_depth, truth_depth, mask=None, zfar=180.):  # Change zfar value?
        batch_size = pred_depth.shape[0]
        height, width = pred_depth.shape[-2], pred_depth.shape[-1]

        difference = pred_depth - truth_depth
        if mask is not None:
            difference = difference * mask

        loss = torch.linalg.norm(difference.view(batch_size, -1), dim=-1, ord=1)
        loss = loss / (height * width * zfar)
        # loss = loss * 0.001

        return torch.mean(loss)

    return depth_loss


def compute_image_gradient(x):
    # idea from tf.image.image_gradients(image)
    # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
    # x: (b,c,h,w), float32 or float64
    # dx, dy: (b,c,h,w)

    # gradient step=1
    left = x
    right = torch.nn.functional.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = torch.nn.functional.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
    dx, dy = right - left, bottom - top
    # dx will always have zeros in the last column, right-left
    # dy will always have zeros in the last row,    bottom-top
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy


# def get_regularity_loss_fn(params):
#     """
#     Returns the following function.
#     :param params:
#     :return:
#     """
#     def regularity_loss(pred_disp, images):
#         batch_size, height, width = pred_disp.shape[0], pred_disp.shape[-2], pred_disp.shape[-1]
#         normalized_disp = pred_disp / torch.mean(pred_disp.view(batch_size, -1),
#                                                  dim=1
#                                                  ).view(batch_size, 1, 1, 1
#                                                         ).expand(-1, -1, height, width)
#
#         dx_disp, dy_disp = compute_image_gradient(normalized_disp)
#         dx_images, dy_images = compute_image_gradient(images)
#
#         dx_disp = dx_disp.view(batch_size, -1)
#         dy_disp = dy_disp.view(batch_size, -1)
#         dx_images = dx_images.view(batch_size, -1)
#         dy_images = dy_images.view(batch_size, -1)
#
#         M1 = 1000.  # 1. ; height * width
#         M2 = height * width  # 1
#
#         loss = torch.linalg.norm(dx_disp/M1, dim=-1, ord=1) * torch.exp(-torch.linalg.norm(dx_images/M2, dim=-1, ord=1)) + \
#                torch.linalg.norm(dy_disp/M1, dim=-1, ord=1) * torch.exp(-torch.linalg.norm(dy_images/M2, dim=-1, ord=1))
#
#         return torch.mean(loss)
#
#     return regularity_loss


def get_regularity_loss_fn(params):
    """
    Returns the following function.
    :param params:
    :return:
    """
    def regularity_loss_fn(disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        Credits to https://github.com/nianticlabs/manydepth/blob/master/manydepth/
        """
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    return regularity_loss_fn


def regularity_tab(disp, img):
    """Computes the smoothness loss per pixel for a disparity image
    The color image is used for edge-aware smoothness
    Credits to https://github.com/nianticlabs/manydepth/blob/master/manydepth/
    """
    height, width = disp.shape[-2], disp.shape[-1]
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x[:, :, :height-2, :width-2] + grad_disp_y[:, :, :height-2, :width-2]


def get_ssim_loss_fn(params):
    """
    Returns SSIM loss function.
    Credits to https://github.com/nianticlabs/manydepth/blob/master/manydepth/

    :param params: (Params) Parameters file.
    :return: (SSIM) SSIM loss function.
    """

    return SSIM()


def get_reconstruction_loss_fn(params):
    """

    :param params:
    :return:
    """
    def reconstruction_loss_fn(params,
                               input_images, input_alpha_images, mask,
                               cameras, alpha_cameras,
                               predicted_depth,
                               depth_model,
                               ssim_loss_fn,
                               channel_is_at_the_end=True,
                               padding_mode='border'):
        """

        :param params:
        :param input_images:
        :param input_alpha_images:
        :param cameras:
        :param alpha_cameras:
        :param predicted_depth:
        :param depth_model:
        :param ssim_loss_fn:
        :param channel_is_at_the_end:
        :param padding_mode: (str) Padding mode for grid sampling during warping.
        Can be 'zeros', 'border' or 'reflection'.
        :return:
        """
        batch_size = input_images.shape[0]
        n_alpha = input_alpha_images.shape[1]
        if channel_is_at_the_end:
            height, width, n_channels = input_images.shape[-3], input_images.shape[-2], input_images.shape[-1]
            images = input_images
            alpha_images = input_alpha_images
        else:
            n_channels, height, width = input_images.shape[-3], input_images.shape[-2], input_images.shape[-1]
            images = transpose_channels(input_images, channel_is_at_the_end=False)
            alpha_images = transpose_channels(input_alpha_images, channel_is_at_the_end=False)

        # Warping operation
        if params.jz or params.ddp:
            cost_volume_builder = depth_model.module.depth_decoder.cost_volume_builder
        else:
            cost_volume_builder = depth_model.depth_decoder.cost_volume_builder

        if params.use_mask:
            warp_depth = 0. + predicted_depth
            warp_depth[~mask] = params.zfar
        else:
            warp_depth = predicted_depth

        world_points = cost_volume_builder.reproject_depth_map(warp_depth, cameras)
        world_points = world_points.view(batch_size, 1, height, width, n_channels).expand(-1, n_alpha, -1, -1, -1)

        warped_images = cost_volume_builder.warp(
            target_world_points=world_points.contiguous().view(-1, height, width, n_channels),
            source_features=alpha_images.view(-1, height, width, n_channels),
            source_cameras=alpha_cameras,
            features_channel_is_at_the_end=True,
            mode='bilinear',
            resize_target_to_fit_source=False,
            padding_mode=padding_mode).view(batch_size, n_alpha,
                                            height, width, n_channels)

        expanded_images = images.view(batch_size, 1, height, width, n_channels).expand(-1, n_alpha, -1, -1, -1)

        # Computing L1-loss
        l1_loss = torch.abs(expanded_images - warped_images).mean(-1, keepdim=True)

        # Computing, if needed, SSIM loss:
        if params.ssim_factor > 0:
            ssim_loss = ssim_loss_fn(
                transpose_channels(expanded_images.contiguous().view(-1, height, width, 3), channel_is_at_the_end=True),
                transpose_channels(warped_images.view(-1, height, width, 3), channel_is_at_the_end=True))

            ssim_loss = transpose_channels(ssim_loss, channel_is_at_the_end=False
                                           ).view(batch_size, n_alpha, height, width, n_channels
                                                  ).mean(-1, keepdim=True)

            loss = params.ssim_factor * ssim_loss + (1-params.ssim_factor) * l1_loss
        else:
            loss = l1_loss

        # Taking min over warped frames:
        loss = torch.min(loss, dim=1, keepdim=False)[0]

        # If using mask, we average only on masked pixels:
        if params.use_mask:
            mask_factor = mask.sum(1, keepdim=True).sum(2, keepdim=True).expand(-1, height, width, -1) + 1e-7
            loss = torch.sum(loss * mask / mask_factor)
        else:
            loss = torch.mean(loss)

        return loss

    return reconstruction_loss_fn


# def get_ssim_loss_fn(params):
#     """
#     Returns SSIM loss function.
#     Credits to https://github.com/VainF/pytorch-msssim
#
#     :param params: (Params) Parameters file.
#     :return: (SSIM) SSIM loss function.
#     """
#
#     return SSIMmap(data_range=1., win_size=params.ssim_window_size, win_sigma=params.ssim_sigma)


# def get_reconstruction_loss_fn(params):
#     """
#
#     :param params:
#     :return:
#     """
#     def reconstruction_loss_fn(params,
#                                input_images, input_alpha_images, mask,
#                                cameras, alpha_cameras,
#                                predicted_depth,
#                                depth_model,
#                                ssim_loss_fn,
#                                channel_is_at_the_end=True,
#                                padding_mode='border'):
#         """
#
#         :param params:
#         :param input_images:
#         :param input_alpha_images:
#         :param cameras:
#         :param alpha_cameras:
#         :param predicted_depth:
#         :param depth_model:
#         :param ssim_loss_fn:
#         :param channel_is_at_the_end:
#         :param padding_mode: (str) Padding mode for grid sampling during warping.
#         Can be 'zeros', 'border' or 'reflection'.
#         :return:
#         """
#         batch_size = input_images.shape[0]
#         n_alpha = input_alpha_images.shape[1]
#         if channel_is_at_the_end:
#             height, width, n_channels = input_images.shape[-3], input_images.shape[-2], input_images.shape[-1]
#             images = input_images
#             alpha_images = input_alpha_images
#         else:
#             n_channels, height, width = input_images.shape[-3], input_images.shape[-2], input_images.shape[-1]
#             images = transpose_channels(input_images, channel_is_at_the_end=False)
#             alpha_images = transpose_channels(input_alpha_images, channel_is_at_the_end=False)
#
#         if params.jz or params.ddp:
#             cost_volume_builder = depth_model.module.depth_decoder.cost_volume_builder
#         else:
#             cost_volume_builder = depth_model.depth_decoder.cost_volume_builder
#
#         if params.use_mask:
#             warp_depth = 0. + predicted_depth
#             warp_depth[~mask] = params.zfar
#         else:
#             warp_depth = predicted_depth
#
#         world_points = cost_volume_builder.reproject_depth_map(warp_depth, cameras)
#         world_points = world_points.view(batch_size, 1, height, width, n_channels).expand(-1, n_alpha, -1, -1, -1)
#
#         warped_images = cost_volume_builder.warp(
#             target_world_points=world_points.contiguous().view(-1, height, width, n_channels),
#             source_features=alpha_images.view(-1, height, width, n_channels),
#             source_cameras=alpha_cameras,
#             features_channel_is_at_the_end=True,
#             mode='bilinear',
#             resize_target_to_fit_source=False,
#             padding_mode=padding_mode).view(batch_size, n_alpha,
#                                             height, width, n_channels)
#
#         expanded_images = images.view(batch_size, 1, height, width, n_channels).expand(-1, n_alpha, -1, -1, -1)
#
#         # Computing L1-loss
#         l1_loss = torch.abs(expanded_images - warped_images).mean(-1)
#         if params.ssim_factor > 0:
#             ssim_loss = ssim_loss_fn(
#                 transpose_channels(expanded_images, channel_is_at_the_end=True).view(-1, 3, height, width),
#                 transpose_channels(warped_images, channel_is_at_the_end=True).view(-1, 3, height, width))
#             ssim_loss = (1. - transpose_channels(pad(ssim_loss,
#                                                      padding=params.ssim_padding,
#                                                      padding_mode='reflect'),
#                                                  channel_is_at_the_end=False
#                                                  ).view(batch_size, n_alpha, height, width, n_channels
#                                                         ).mean(-1))
#             all_loss = params.ssim_factor/2. * ssim_loss + (1-params.ssim_factor) * l1_loss
#         else:
#             all_loss = l1_loss
#
#         # loss_argmin = torch.min(all_loss, dim=1)[1].view(batch_size, 1, height, width, 1).expand(-1, -1, -1, -1, n_channels)
#         loss_argmin = torch.min(all_loss, dim=1, keepdim=True)[1]
#         loss = torch.gather(input=all_loss, dim=1, index=loss_argmin)[:, 0]
#
#         if params.use_mask:
#             loss[~mask[..., 0]] = 0.
#
#         return torch.mean(loss)
#
#     return reconstruction_loss_fn


def preprocess_input_dict(input_dict, device):
    images = torch.cat(input_dict['rgb'], dim=0).to(device)
    zbuf = torch.cat(input_dict['zbuf'], dim=0).to(device)
    mask = torch.cat(input_dict['mask'], dim=0).to(device)
    R = torch.cat(input_dict['R'], dim=0).to(device)
    T = torch.cat(input_dict['T'], dim=0).to(device)
    zfar = torch.Tensor(input_dict['zfar']).to(device)
    path = input_dict['path']
    index = input_dict['index']

    return images, zbuf, mask, R, T, zfar, path, index


# TO CHANGE! RIGHT NOW, THIS FUNCTION IS UNDER-OPTIMIZED
def gather_neighbor_frames(dataset, index, alphas, device):
    neighbor_images, neighbor_zbuf, neighbor_mask = [], [], []
    neighbor_R, neighbor_T, neighbor_zfar = [], [], []

    for i in range(len(index)):
        alpha_images, alpha_zbuf, alpha_mask = [], [], []
        alpha_R, alpha_T, alpha_zfar = [], [], []

        for alpha in alphas:
            frame_index = index[i]
            # We load the neighbor frame for alpha
            neighbor_frame = dataset.get_neighbor_frame_from_idx(frame_index, alpha, device='cpu')

            # We process the dictionary
            images = neighbor_frame['rgb'].to(device)
            zbuf = neighbor_frame['zbuf'].to(device)
            mask = neighbor_frame['mask'].to(device)
            R = neighbor_frame['R'].to(device)
            T = neighbor_frame['T'].to(device)
            zfar = torch.Tensor([neighbor_frame['zfar']]).to(device)

            # We gather this neighbor frame with other neighbor frames that have a different alpha value
            alpha_images.append(images), alpha_zbuf.append(zbuf), alpha_mask.append(mask)
            alpha_R.append(R), alpha_T.append(T), alpha_zfar.append(zfar)

        # We concatenate neighbors with different alpha values
        alpha_images = torch.stack(alpha_images, dim=1)
        alpha_zbuf = torch.stack(alpha_zbuf, dim=1)
        alpha_mask = torch.stack(alpha_mask, dim=1)
        alpha_R = torch.stack(alpha_R, dim=1)
        alpha_T= torch.stack(alpha_T, dim=1)
        alpha_zfar = torch.stack(alpha_zfar, dim=1)

        # Then we gather them with neighbors of other frames
        neighbor_images.append(alpha_images), neighbor_zbuf.append(alpha_zbuf), neighbor_mask.append(alpha_mask)
        neighbor_R.append(alpha_R), neighbor_T.append(alpha_T), neighbor_zfar.append(alpha_zfar)

    # Finally, we concatenate all neighbors on the first dimension
    neighbor_images = torch.cat(neighbor_images, dim=0)
    neighbor_zbuf = torch.cat(neighbor_zbuf, dim=0)
    neighbor_mask = torch.cat(neighbor_mask, dim=0)
    neighbor_R = torch.cat(neighbor_R, dim=0)
    neighbor_T = torch.cat(neighbor_T, dim=0)
    neighbor_zfar = torch.cat(neighbor_zfar, dim=0)

    return neighbor_images, neighbor_zbuf, neighbor_mask, neighbor_R, neighbor_T, neighbor_zfar


def compute_depth_from_disparity(params, disp):
    a = 1./params.znear - 1./params.zfar
    b = 1./params.zfar

    return 1. / (a*disp + b)


def compute_disparity_from_depth(params, depth):
    a = 1./params.znear - 1./params.zfar
    b = 1./params.zfar

    return (1./depth - b) / a


def convert_matrix_to_pose(params, R, T, alpha_R, alpha_T):
    batch_size, n_alpha = T.shape[0], alpha_T.shape[1]

    expanded_R = R.view(batch_size, 1, 3, 3).expand(-1, n_alpha, -1, -1)
    expanded_T = T.view(batch_size, 1, 3).expand(-1, n_alpha, -1)

    relative_R = expanded_R.transpose(dim0=-1, dim1=-2) @ alpha_R
    relative_T = alpha_T - quaternion_apply(matrix_to_quaternion(relative_R.transpose(dim0=-1, dim1=-2)), expanded_T)

    pose_angle = quaternion_to_axis_angle(matrix_to_quaternion(relative_R)) / params.pose_factor
    pose_T = relative_T / params.pose_factor

    return torch.cat((pose_T, pose_angle), dim=-1)


def adjust_image(x, brightness_factor, contrast_factor,
                          saturation_factor, hue_factor):
    res = torchvision.transforms.functional.adjust_brightness(x, brightness_factor)
    res = torchvision.transforms.functional.adjust_contrast(res, contrast_factor)
    res = torchvision.transforms.functional.adjust_saturation(res, saturation_factor)
    res = torchvision.transforms.functional.adjust_hue(res, hue_factor)

    return res


def apply_jitter_to_images(params, x, x_alpha):
    brightness_factor = max(0, 1 + params.brightness_jitter_range * (1 - 2*np.random.rand()))
    contrast_factor = max(0, 1 + params.contrast_jitter_range * (1 - 2 * np.random.rand()))
    saturation_factor = max(0, 1 + params.saturation_jitter_range * (1 - 2 * np.random.rand()))
    hue_factor = params.hue_jitter_range * (1 - 2 * np.random.rand())

    adjusted_x = adjust_image(x, brightness_factor, contrast_factor, saturation_factor, hue_factor)
    adjusted_x_alpha = adjust_image(x_alpha, brightness_factor, contrast_factor, saturation_factor, hue_factor)

    return adjusted_x, adjusted_x_alpha


def apply_symmetry_to_images(x, zbuf, R, T, mask=None):
    T_flip = 0. + T
    T_flip[..., 0] = -1. * T_flip[..., 0]

    # In view space, the rotation is flipped on X and Z axis
    R_flip = matrix_to_euler_angles(R, convention='XYZ')
    R_flip[..., 1], R_flip[..., 2] = -1. * R_flip[..., 1], -1. * R_flip[..., 2]
    R_flip = euler_angles_to_matrix(R_flip, convention='XYZ')

    flipped_x = hflip(x)

    if zbuf is not None:
        flipped_zbuf = hflip(transpose_channels(zbuf, channel_is_at_the_end=True))
        flipped_zbuf = transpose_channels(flipped_zbuf, channel_is_at_the_end=False)
    else:
        flipped_zbuf = None

    if mask is not None:
        flipped_mask = hflip(transpose_channels(mask, channel_is_at_the_end=True))
        flipped_mask = transpose_channels(flipped_mask, channel_is_at_the_end=False).bool()

        return flipped_x, flipped_zbuf, R_flip, T_flip, flipped_mask

    else:
        return flipped_x, flipped_zbuf, R_flip, T_flip
