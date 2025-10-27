import copy
import json
import os.path as osp
import queue
import threading
from PIL import Image
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import torch
import os
import numpy as np
import glob
from copy import deepcopy
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import sys
from pathlib import Path
import copy
import os
import torch.nn.functional as F
from einops import rearrange
from typing import List, Dict, Tuple, Any, Optional, Literal
def umeyama_alignment(source, target, with_scaling=True):
    """
    Implements the Umeyama algorithm for point cloud alignment.
    
    Args:
        source: Source point cloud of shape [n, 3]
        target: Target point cloud of shape [n, 3]
        with_scaling: Boolean flag to determine if scaling should be applied
        
    Returns:
        R: Rotation matrix [3, 3]
        t: Translation vector [3]
        s: Scaling factor (1.0 if with_scaling=False)
    """
    # Get device from input tensors
    device = source.device
    
    # Compute means
    source_mean = torch.mean(source, dim=0)
    target_mean = torch.mean(target, dim=0)
    
    # Center both point clouds
    source_centered = source - source_mean
    target_centered = target - target_mean
    
    # Number of points
    n = source.shape[0]
    
    # Covariance matrix
    cov = torch.matmul(target_centered.T, source_centered) / n
    
    # SVD decomposition
    U, D, Vh = torch.linalg.svd(cov)
    V = Vh.T
    
    # Check if rotation matrix might need correction (for right-handedness)
    det_UV = torch.det(torch.matmul(U, V.T))
    S = torch.eye(3, device=device)
    if det_UV < 0:
        S[-1, -1] = -1
    
    # Rotation matrix
    R = torch.matmul(U, torch.matmul(S, V.T))
    
    # Scaling factor
    if with_scaling:
        source_var = torch.sum(torch.square(source_centered)) / n
        s = torch.sum(D * S.diag()) / source_var
    else:
        s = torch.tensor(1.0, device=device)
    
    # Translation vector - ensures source transforms to target
    t = target_mean - s * torch.matmul(source_mean, R.T)
    
    return R, t, s

def find_alignment_transform(source_points, target_points, with_scaling=True):
    """
    计算从source到target的变换参数，不修改原始点云
    
    Args:
        source_points: 源点云 [n, 3]
        target_points: 目标点云 [n, 3]
        with_scaling: 是否应用缩放
        
    Returns:
        valid_mask: 有效点的掩码
        R: 旋转矩阵
        t: 平移向量
        s: 缩放因子
    """
    # # 找出有效点 (不含NaN的点)
    # valid_mask = ~torch.isnan(source_points).any(dim=1) & ~torch.isnan(target_points).any(dim=1)
    # valid_source = source_points[valid_mask]
    # valid_target = target_points[valid_mask]
    
    # # 至少需要3个点才能进行对齐
    # if len(valid_source) < 3:
    #     raise ValueError(f"需要至少3个有效点进行对齐，但只有{len(valid_source)}个")
    
    # 计算变换参数
    R, t, s = umeyama_alignment(source_points, target_points, with_scaling)
    
    return R, t, s

def apply_transform(points, R, t, s=1.0):
    """
    应用变换到点云
    
    Args:
        points: 点云 [n, 3]
        R: 旋转矩阵 [3, 3]
        t: 平移向量 [3]
        s: 缩放因子
        
    Returns:
        transformed_points: 变换后的点云 [n, 3]
    """
    # 确保输入是浮点类型
    points = points.float()
    # 应用变换: 缩放 -> 旋转 -> 平移
    transformed_points = s * torch.matmul(points, R.T) + t
    return transformed_points



def to_homogeneous_matrix(poses):
    """Convert 3x4 pose matrix to 4x4 homogeneous matrix"""
    N = poses.shape[0]
    device = poses.device
    dtype = poses.dtype
    
    poses_4x4 = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)
    poses_4x4[:, :3, :] = poses
    
    return poses_4x4

def convert_opencv_to_pytorch3d(poses):
    """Convert poses from OpenCV to PyTorch3D coordinate system"""
    if poses.shape[1] == 3:
        poses = to_homogeneous_matrix(poses)
    
    device = poses.device
    dtype = poses.dtype
    
    conversion = torch.tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], device=device, dtype=dtype)
    
    return conversion @ poses @ conversion.inverse()


def apply_alignment(points, R, t, s):
    """Apply alignment transformation to points"""
    return s * (R @ points.t()).t() + t.t()

def visualize_camera_poses(pred_poses, gt_poses, title="Camera Poses"):
    """Visualize camera poses with orientation
    Args:
        pred_poses: (N, 4, 4) predicted camera poses
        gt_poses: (N, 4, 4) ground truth camera poses
        title: plot title
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot camera centers and axes
    length = 0.5  # Length of coordinate axes
    
    # Define colors for each dataset
    colors = {'pred': 'red', 'gt': 'blue'}
    axis_colors = ['red', 'green', 'blue']  # Colors for X, Y, Z axes
    
    for poses, pose_type in [(pred_poses, 'pred'), (gt_poses, 'gt')]:
        # Convert to numpy and CPU
        poses_np = poses.cpu().numpy()
        centers = poses_np[:, :3, 3]
        
        # Plot trajectory
        color = colors[pose_type]
        label = 'Predicted' if pose_type == 'pred' else 'Ground Truth'
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], 
                color=color, linestyle='-', label=label)
        
        # Plot camera positions
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
                  color=color, marker='o', s=20)
        
        # Plot coordinate axes for each camera
        for pose in poses_np:
            center = pose[:3, 3]
            
            # Plot coordinate axes
            for i in range(3):
                axis = pose[:3, i]  # Get axis direction
                endpoint = center + length * axis
                ax.plot([center[0], endpoint[0]],
                       [center[1], endpoint[1]],
                       [center[2], endpoint[2]],
                       color=axis_colors[i], linestyle='-', linewidth=1)
    
    # Set equal aspect ratio
    max_range = np.array([
        pred_poses[:, :3, 3].cpu().numpy().ptp(axis=0).max(),
        gt_poses[:, :3, 3].cpu().numpy().ptp(axis=0).max()
    ]).max() / 2.0
    
    mid = (pred_poses[:, :3, 3].mean(dim=0).cpu().numpy() + 
           gt_poses[:, :3, 3].mean(dim=0).cpu().numpy()) / 2
    
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    plt.show()


def align_poses_with_orientation(pred_poses, gt_poses):
    """Align predicted trajectory to ground truth considering camera orientation
    Args:
        pred_poses: (N, 4, 4) predicted camera poses
        gt_poses: (N, 4, 4) ground truth camera poses
    Returns:
        R: (3, 3) rotation matrix
        t: (3, 1) translation vector
        s: scale factor
    """
    device = pred_poses.device
    dtype = pred_poses.dtype
    N = pred_poses.shape[0]

    # Extract rotation matrices and translations
    pred_R = pred_poses[:, :3, :3].contiguous()  # (N, 3, 3)
    pred_t = pred_poses[:, :3, 3].contiguous()   # (N, 3)
    gt_R = gt_poses[:, :3, :3].contiguous()      # (N, 3, 3)
    gt_t = gt_poses[:, :3, 3].contiguous()       # (N, 3)

    # Center the translations
    pred_t_mean = pred_t.mean(dim=0)  # (3,)
    gt_t_mean = gt_t.mean(dim=0)      # (3,)
    pred_t_centered = pred_t - pred_t_mean
    gt_t_centered = gt_t - gt_t_mean

    # Stack positions and orientations
    pred_data = torch.cat([
        pred_t_centered,                    # (N, 3)
        pred_R.reshape(N, 9)                # (N, 9)
    ], dim=1)  # (N, 12)
    
    gt_data = torch.cat([
        gt_t_centered,                      # (N, 3)
        gt_R.reshape(N, 9)                  # (N, 9)
    ], dim=1)  # (N, 12)

    # Compute optimal rotation
    H = pred_data.t() @ gt_data  # (12, 12)
    U, S, Vh = torch.linalg.svd(H[:3, :3])  # Only use position for initial alignment
    R = Vh.t() @ U.t()

    # Ensure proper rotation matrix (det=1)
    det = torch.det(R)
    print(f"Rotation matrix determinant: {det.item()}")
    if det < 0:
        Vh[-1] *= -1
        R = Vh.t() @ U.t()

    # Compute scale using positions
    pred_var = (pred_t_centered ** 2).sum(dim=1).mean()
    gt_var = (gt_t_centered ** 2).sum(dim=1).mean()
    s = torch.sqrt(gt_var / pred_var)

    # Compute translation
    t = gt_t_mean.unsqueeze(1) - s * (R @ pred_t_mean.unsqueeze(1))

    print("\nAlignment Parameters:")
    print(f"Scale: {s.item():.4f}")
    print(f"Rotation:\n{R.cpu().numpy()}")
    print(f"Translation:\n{t.cpu().numpy()}")
    
    # Verify alignment quality using both position and orientation
    aligned_pred_data = torch.cat([
        s * (R @ pred_t_centered.t()).t(),     # Transform positions
        (R @ pred_R.reshape(N, 3, 3).transpose(1, 2)).transpose(1, 2).reshape(N, 9)  # Transform orientations
    ], dim=1)
    
    error = torch.norm(aligned_pred_data - gt_data, dim=1)
    print(f"\nAlignment error statistics:")
    print(f"Mean error: {error.mean():.4f}")
    print(f"Std error: {error.std():.4f}")
    
    # Compute orientation error before and after alignment
    ori_error_before = torch.arccos(torch.clamp(
        (pred_R.reshape(N, 9) * gt_R.reshape(N, 9)).sum(dim=1) / 3,
        -1.0, 1.0
    )) * 180 / np.pi
    
    aligned_R = (R @ pred_R.reshape(N, 3, 3)).reshape(N, 9)
    ori_error_after = torch.arccos(torch.clamp(
        (aligned_R * gt_R.reshape(N, 9)).sum(dim=1) / 3,
        -1.0, 1.0
    )) * 180 / np.pi
    
    print(f"\nOrientation errors (degrees):")
    print(f"Before alignment: {ori_error_before.mean():.4f} ± {ori_error_before.std():.4f}")
    print(f"After alignment: {ori_error_after.mean():.4f} ± {ori_error_after.std():.4f}")

    return R, t, s

def align_trajectory_and_pointcloud(pred_poses, gt_poses, pred_points, device=None):
    """Main function to align predicted trajectory and point cloud to ground truth"""
    # Move to device and ensure float32 dtype
    device = device or pred_poses.device
    dtype = torch.float32
    
    pred_poses = pred_poses.to(device).to(dtype)
    gt_poses = gt_poses.to(device).to(dtype)
    pred_points = pred_points.to(device).to(dtype)
    
    print(f"\nInitial shapes and types:")
    print(f"Predicted poses: {pred_poses.shape}, {pred_poses.dtype}")
    print(f"Ground truth poses: {gt_poses.shape}, {gt_poses.dtype}")
    print(f"Predicted points: {pred_points.shape}, {pred_points.dtype}")
    
    # Convert predicted poses to PyTorch3D coordinate system
    pred_poses = convert_opencv_to_pytorch3d(pred_poses)
    print(f"Poses after coordinate conversion: {pred_poses.shape}")
    
    # Align trajectories considering orientation
    R, t, s = align_poses_with_orientation(pred_poses, gt_poses)
    
    # Apply alignment to poses
    N = pred_poses.shape[0]
    aligned_poses = pred_poses.clone()
    for i in range(N):
        # Apply to rotation
        aligned_poses[i, :3, :3] = s * R @ pred_poses[i, :3, :3]
        # Apply to translation
        aligned_poses[i, :3, 3] = s * (R @ pred_poses[i, :3, 3]) + t.squeeze()
    
    # Apply alignment to point cloud
    aligned_points = s * (R @ pred_points.t()).t() + t.t()
    
    # Compute pose errors
    pos_error_before = torch.norm(pred_poses[:, :3, 3] - gt_poses[:, :3, 3], dim=1)
    pos_error_after = torch.norm(aligned_poses[:, :3, 3] - gt_poses[:, :3, 3], dim=1)
    
    print(f"\nPosition errors:")
    print(f"Before alignment: {pos_error_before.mean():.4f} ± {pos_error_before.std():.4f}")
    print(f"After alignment: {pos_error_after.mean():.4f} ± {pos_error_after.std():.4f}")
    
    return aligned_poses, aligned_points
##########################################################################
######## The following code is for reconstruction: vggt ################
##########################################################################

from vggt.vggt.models.vggt import VGGT
from vggt.vggt.utils.load_fn import load_and_preprocess_images
from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.vggt.utils.geometry import depth_to_world_coords_points


def load_vggt_model(device) -> VGGT:
    """
    根据已存在的代码，加载并初始化VGGT模型。
    注意此函数假设您已经在 ~/.cache/torch/hub/checkpoints/ 中有model.pt文件
    或者会自动从huggingface下载。

    Returns:
        model: 初始化好的VGGT模型
    """
    print("Initializing and loading VGGT model...")
    
    # 初始化模型 # model.load_state_dict(torch.load("/home/sli/phd_projects/vggt/training/logs/custom_dataset_finetune/ckpts/best_model.pt")['model'],strict=False)

    model = VGGT()
    
    # 加载预训练权重
    # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.load_state_dict(torch.load("/home/sli/phd_projects/vggt-robotics/training/logs_all_2m_loss/exp001/ckpts/checkpoint.pt")['model'],strict=False)
    
    # 设置为评估模式并移至适当设备
    model.eval()
    model = model.to(device)
    
    print("VGGT model loaded successfully")
    return model

# only keep the points < 2m

def predict_pointcloud_from_images(image_paths: List[str], model, device: str, 
                                 pixel_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    从图像预测点云并获取指定像素的3D坐标
    
    Args:
        image_paths: 图像路径列表
        model: VGGT模型  
        device: 设备
        pixel_coords: 像素坐标 (N, 2)
        
    Returns:
        pred_points: 预测点云 (M, 3)
        pred_colors: 点云颜色 (M, 3)  
        specific_points_list: 每张图像对应像素的3D坐标列表
    """
    if len(image_paths) == 0:
        raise ValueError("图像路径列表为空")
    
    print(f"处理 {len(image_paths)} 张图像进行点云预测")
    
    # 加载和预处理图像 - 使用多坐标版本
    images, coords_list = load_and_preprocess_images(image_paths, pixel_coords=pixel_coords)
    images = images.to(device)
    
    # 将坐标移到设备上
    for i in range(len(coords_list)):
        if coords_list[i] is not None:
            coords_list[i] = coords_list[i].to(device)
    
    print(f"预处理后的图像形状: {images.shape}")
    
    # 根据GPU能力设置精度
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    
    # 运行推理
    print("运行推理...")
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda', dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)
    # print(predictions.keys()) 
    # # print(predictions['world_points'].shape)
    # torch.tensor()
    # 将姿态编码转换为外部和内部矩阵
    print("将姿态编码转换为相机参数...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # 获取深度图
    print("从深度图计算世界点...")
    depth_map = predictions["depth"]
    
    # 移除额外的批次维度
    if depth_map.dim() == 5 and depth_map.shape[0] == 1:
        depth_map = depth_map.squeeze(0)
    
    if extrinsic.dim() == 4 and extrinsic.shape[0] == 1:
        extrinsic = extrinsic.squeeze(0)
    
    if intrinsic.dim() == 4 and intrinsic.shape[0] == 1:
        intrinsic = intrinsic.squeeze(0)
    
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Extrinsic shape: {extrinsic.shape}")
    print(f"Intrinsic shape: {intrinsic.shape}")
    
    # 计算世界坐标点
    world_points_list = []
    num_frames = depth_map.shape[0]
    
    # 转换为numpy数组
    depth_map_np = depth_map.detach().cpu().numpy()
    extrinsic_np = extrinsic.detach().cpu().numpy()
    intrinsic_np = intrinsic.detach().cpu().numpy()
    
    # 对每一帧处理
    for frame_idx in range(num_frames):
        # 获取当前帧的深度图，确保它是2D的
        cur_depth = depth_map_np[frame_idx]
        if cur_depth.ndim > 2:
            cur_depth = cur_depth.squeeze(-1)
        
        # 截断深度图：只保留距离<=2m的点
        cur_depth = np.where(cur_depth <= 1.8, cur_depth, 0)
        
        # 获取当前帧的相机参数
        cur_extrinsic = extrinsic_np[frame_idx]
        cur_intrinsic = intrinsic_np[frame_idx]
        
        # 确保intrinsic是(3,3)形状
        if cur_intrinsic.shape != (3, 3):
            print(f"警告: 帧{frame_idx} Intrinsic形状不正确: {cur_intrinsic.shape}")
            if cur_intrinsic.shape == (3, 4):
                cur_intrinsic = cur_intrinsic[:, :3]
                print(f"修正后的Intrinsic形状: {cur_intrinsic.shape}")
            else:
                print(f"跳过帧 {frame_idx}，intrinsic形状无法修正")
                continue
        
        # 调用depth_to_world_coords_points函数
        try:
            cur_world_points, _, _ = depth_to_world_coords_points(
                cur_depth, cur_extrinsic, cur_intrinsic
            )
            world_points_list.append(cur_world_points)
            print(f"帧 {frame_idx}: 成功生成世界坐标点 {cur_world_points.shape} (<=2m only)")
        except Exception as e:
            print(f"处理帧 {frame_idx} 时出错: {e}")
            continue
    
    # 如果没有成功处理任何帧，返回空张量
    if len(world_points_list) == 0:
        print("警告: 没有成功处理任何帧")
        return torch.zeros((0, 3)), torch.zeros((0, 3)), []
    
    # 将所有帧的世界坐标点堆叠起来
    world_points_np = np.stack(world_points_list, axis=0)
    world_points = torch.from_numpy(world_points_np).to(device)
    
    # 从原始图像中提取每个点的颜色
    print("提取点云颜色...")
    all_points = []
    all_colors = []
    specific_points_list = []
    
    # 为特定像素坐标创建结果列表
    for frame_idx in range(len(image_paths)):
        specific_points_list.append(torch.full((pixel_coords.shape[0], 3), float('nan'), device=device))
    
    for i, image_path in enumerate(image_paths):
        # 跳过超出范围的索引
        if i >= world_points.shape[0]:
            print(f"跳过图像 {i}，超出world_points范围")
            continue
            
        # 获取当前帧的点云
        frame_points = world_points[i]  # (H, W, 3)
        H, W, _ = frame_points.shape
        
        # 加载原始图像以获取颜色
        try:
            # 加载图像
            img = Image.open(image_path)
            img_np = np.array(img)
            
            # 确保图像是RGB
            if len(img_np.shape) == 2:  # 灰度图
                img_np = np.stack([img_np, img_np, img_np], axis=2)
            elif img_np.shape[2] == 4:  # RGBA
                img_np = img_np[:, :, :3]
                
            # 转换为PyTorch张量
            img_tensor = torch.from_numpy(img_np).float().to(device) / 255.0
            
            # 调整图像大小以匹配点云
            if img_tensor.shape[0] != H or img_tensor.shape[1] != W:
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor.permute(2, 0, 1).unsqueeze(0),  # [1, C, H, W]
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0)  # [H, W, C]
            
            # 获取当前图像的特定像素坐标对应的世界坐标
            frame_coords = coords_list[i] if i < len(coords_list) else None
            if frame_coords is not None:
                for j in range(frame_coords.shape[0]):
                    if j < pixel_coords.shape[0]:  # 确保索引不越界
                        y, x = frame_coords[j].long()
                        # 检查坐标是否在范围内
                        if 0 <= y < H and 0 <= x < W:
                            specific_points_list[i][j] = frame_points[y, x]
            
            # 展平点云和颜色
            flat_points = frame_points.reshape(-1, 3)
            flat_colors = img_tensor.reshape(-1, 3)
            
            # 创建掩码数组，标记需要保留的点
            mask = torch.ones(flat_points.shape[0], dtype=torch.bool, device=device)
            
            # 基本过滤：移除无效点
            finite_mask = torch.isfinite(flat_points).all(dim=1)
            valid_norm_mask = torch.norm(flat_points, dim=1) > 1e-6
            mask = mask & finite_mask & valid_norm_mask
            
            # 应用置信度阈值（如果可用）
            if "depth_conf" in predictions:
                confidence = predictions["depth_conf"]
                if isinstance(confidence, torch.Tensor):
                    # 确保有正确的形状
                    if confidence.dim() == 4 and confidence.shape[0] == 1:
                        confidence = confidence.squeeze(0)  # 移除批次维度
                    
                    if i < confidence.shape[0]:
                        conf_frame = confidence[i]
                        conf_flat = conf_frame.reshape(-1)
                        # 确保维度匹配
                        if conf_flat.shape[0] == mask.shape[0]:
                            conf_mask = conf_flat > 0.8# 根据需要调整阈值
                            mask = mask & conf_mask
            
            # 应用掩码
            flat_points = flat_points[mask]
            flat_colors = flat_colors[mask]
            
            if flat_points.shape[0] > 0:
                all_points.append(flat_points)
                all_colors.append(flat_colors)
                print(f"图像 {i}: 添加了 {flat_points.shape[0]} 个有效点")
            else:
                print(f"图像 {i}: 没有有效点")
            
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            continue
    
    # 如果没有有效点，返回空张量
    if len(all_points) == 0:
        print("警告: 没有生成任何有效点")
        return torch.zeros((0, 3)), torch.zeros((0, 3)), specific_points_list
    
    # 组合所有点和颜色
    points = torch.cat(all_points, dim=0)
    colors = torch.cat(all_colors, dim=0)
    
    print(f"生成的点云有 {points.shape[0]} 个点 (<=2m only)")
    
    return points, colors, specific_points_list

# predict points from depth map 

# def predict_pointcloud_from_images(image_paths: List[str], model, device: str, 
#                                  pixel_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
#     """
#     从图像预测点云并获取指定像素的3D坐标
    
#     Args:
#         image_paths: 图像路径列表
#         model: VGGT模型  
#         device: 设备
#         pixel_coords: 像素坐标 (N, 2)
        
#     Returns:
#         pred_points: 预测点云 (M, 3)
#         pred_colors: 点云颜色 (M, 3)  
#         specific_points_list: 每张图像对应像素的3D坐标列表
#     """
#     if len(image_paths) == 0:
#         raise ValueError("图像路径列表为空")
    
#     print(f"处理 {len(image_paths)} 张图像进行点云预测")
    
#     # 加载和预处理图像 - 使用多坐标版本
#     images, coords_list = load_and_preprocess_images(image_paths, pixel_coords=pixel_coords)
#     images = images.to(device)
    
#     # 将坐标移到设备上
#     for i in range(len(coords_list)):
#         if coords_list[i] is not None:
#             coords_list[i] = coords_list[i].to(device)
    
#     print(f"预处理后的图像形状: {images.shape}")
    
#     # 根据GPU能力设置精度
#     if torch.cuda.is_available():
#         dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
#     else:
#         dtype = torch.float32
    
#     # 运行推理
#     print("运行推理...")
#     with torch.no_grad():
#         if torch.cuda.is_available():
#             with torch.amp.autocast('cuda', dtype=dtype):
#                 predictions = model(images)
#         else:
#             predictions = model(images)
#     # print(predictions.keys()) 
#     # # print(predictions['world_points'].shape)
#     # torch.tensor()
#     # 将姿态编码转换为外部和内部矩阵
#     print("将姿态编码转换为相机参数...")
#     extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
#     predictions["extrinsic"] = extrinsic
#     predictions["intrinsic"] = intrinsic

#     # 获取深度图
#     print("从深度图计算世界点...")
#     depth_map = predictions["depth"]
    
#     # 移除额外的批次维度
#     if depth_map.dim() == 5 and depth_map.shape[0] == 1:
#         depth_map = depth_map.squeeze(0)
    
#     if extrinsic.dim() == 4 and extrinsic.shape[0] == 1:
#         extrinsic = extrinsic.squeeze(0)
    
#     if intrinsic.dim() == 4 and intrinsic.shape[0] == 1:
#         intrinsic = intrinsic.squeeze(0)
    
#     print(f"Depth map shape: {depth_map.shape}")
#     print(f"Extrinsic shape: {extrinsic.shape}")
#     print(f"Intrinsic shape: {intrinsic.shape}")
    
#     # 计算世界坐标点
#     world_points_list = []
#     num_frames = depth_map.shape[0]
    
#     # 转换为numpy数组
#     depth_map_np = depth_map.detach().cpu().numpy()
#     extrinsic_np = extrinsic.detach().cpu().numpy()
#     intrinsic_np = intrinsic.detach().cpu().numpy()
    
#     # 对每一帧处理
#     for frame_idx in range(num_frames):
#         # 获取当前帧的深度图，确保它是2D的
#         cur_depth = depth_map_np[frame_idx]
#         if cur_depth.ndim > 2:
#             cur_depth = cur_depth.squeeze(-1)
        
#         # 获取当前帧的相机参数
#         cur_extrinsic = extrinsic_np[frame_idx]
#         cur_intrinsic = intrinsic_np[frame_idx]
        
#         # 确保intrinsic是(3,3)形状
#         if cur_intrinsic.shape != (3, 3):
#             print(f"警告: 帧{frame_idx} Intrinsic形状不正确: {cur_intrinsic.shape}")
#             if cur_intrinsic.shape == (3, 4):
#                 cur_intrinsic = cur_intrinsic[:, :3]
#                 print(f"修正后的Intrinsic形状: {cur_intrinsic.shape}")
#             else:
#                 print(f"跳过帧 {frame_idx}，intrinsic形状无法修正")
#                 continue
        
#         # 调用depth_to_world_coords_points函数
#         try:
#             cur_world_points, _, _ = depth_to_world_coords_points(
#                 cur_depth, cur_extrinsic, cur_intrinsic
#             )
#             world_points_list.append(cur_world_points)
#             print(f"帧 {frame_idx}: 成功生成世界坐标点 {cur_world_points.shape}")
#         except Exception as e:
#             print(f"处理帧 {frame_idx} 时出错: {e}")
#             continue
    
#     # 如果没有成功处理任何帧，返回空张量
#     if len(world_points_list) == 0:
#         print("警告: 没有成功处理任何帧")
#         return torch.zeros((0, 3)), torch.zeros((0, 3)), []
    
#     # 将所有帧的世界坐标点堆叠起来
#     world_points_np = np.stack(world_points_list, axis=0)
#     world_points = torch.from_numpy(world_points_np).to(device)
    
#     # 从原始图像中提取每个点的颜色
#     print("提取点云颜色...")
#     all_points = []
#     all_colors = []
#     specific_points_list = []
    
#     # 为特定像素坐标创建结果列表
#     for frame_idx in range(len(image_paths)):
#         specific_points_list.append(torch.full((pixel_coords.shape[0], 3), float('nan'), device=device))
    
#     for i, image_path in enumerate(image_paths):
#         # 跳过超出范围的索引
#         if i >= world_points.shape[0]:
#             print(f"跳过图像 {i}，超出world_points范围")
#             continue
            
#         # 获取当前帧的点云
#         frame_points = world_points[i]  # (H, W, 3)
#         H, W, _ = frame_points.shape
        
#         # 加载原始图像以获取颜色
#         try:
#             # 加载图像
#             img = Image.open(image_path)
#             img_np = np.array(img)
            
#             # 确保图像是RGB
#             if len(img_np.shape) == 2:  # 灰度图
#                 img_np = np.stack([img_np, img_np, img_np], axis=2)
#             elif img_np.shape[2] == 4:  # RGBA
#                 img_np = img_np[:, :, :3]
                
#             # 转换为PyTorch张量
#             img_tensor = torch.from_numpy(img_np).float().to(device) / 255.0
            
#             # 调整图像大小以匹配点云
#             if img_tensor.shape[0] != H or img_tensor.shape[1] != W:
#                 img_tensor = torch.nn.functional.interpolate(
#                     img_tensor.permute(2, 0, 1).unsqueeze(0),  # [1, C, H, W]
#                     size=(H, W),
#                     mode='bilinear',
#                     align_corners=False
#                 ).squeeze(0).permute(1, 2, 0)  # [H, W, C]
            
#             # 获取当前图像的特定像素坐标对应的世界坐标
#             frame_coords = coords_list[i] if i < len(coords_list) else None
#             if frame_coords is not None:
#                 for j in range(frame_coords.shape[0]):
#                     if j < pixel_coords.shape[0]:  # 确保索引不越界
#                         y, x = frame_coords[j].long()
#                         # 检查坐标是否在范围内
#                         if 0 <= y < H and 0 <= x < W:
#                             specific_points_list[i][j] = frame_points[y, x]
            
#             # 展平点云和颜色
#             flat_points = frame_points.reshape(-1, 3)
#             flat_colors = img_tensor.reshape(-1, 3)
            
#             # 创建掩码数组，标记需要保留的点
#             mask = torch.ones(flat_points.shape[0], dtype=torch.bool, device=device)
            
#             # 基本过滤：移除无效点
#             finite_mask = torch.isfinite(flat_points).all(dim=1)
#             valid_norm_mask = torch.norm(flat_points, dim=1) > 1e-6
#             mask = mask & finite_mask & valid_norm_mask
            
#             # 应用置信度阈值（如果可用）
#             if "depth_conf" in predictions:
#                 confidence = predictions["depth_conf"]
#                 if isinstance(confidence, torch.Tensor):
#                     # 确保有正确的形状
#                     if confidence.dim() == 4 and confidence.shape[0] == 1:
#                         confidence = confidence.squeeze(0)  # 移除批次维度
                    
#                     if i < confidence.shape[0]:
#                         conf_frame = confidence[i]
#                         conf_flat = conf_frame.reshape(-1)
#                         # 确保维度匹配
#                         if conf_flat.shape[0] == mask.shape[0]:
#                             conf_mask = conf_flat > 0.8# 根据需要调整阈值
#                             mask = mask & conf_mask
            
#             # 应用掩码
#             flat_points = flat_points[mask]
#             flat_colors = flat_colors[mask]
            
#             if flat_points.shape[0] > 0:
#                 all_points.append(flat_points)
#                 all_colors.append(flat_colors)
#                 print(f"图像 {i}: 添加了 {flat_points.shape[0]} 个有效点")
#             else:
#                 print(f"图像 {i}: 没有有效点")
            
#         except Exception as e:
#             print(f"处理图像 {image_path} 时出错: {e}")
#             continue
    
#     # 如果没有有效点，返回空张量
#     if len(all_points) == 0:
#         print("警告: 没有生成任何有效点")
#         return torch.zeros((0, 3)), torch.zeros((0, 3)), specific_points_list
    
#     # 组合所有点和颜色
#     points = torch.cat(all_points, dim=0)
#     colors = torch.cat(all_colors, dim=0)
    
#     print(f"生成的点云有 {points.shape[0]} 个点")
    
#     return points, colors, specific_points_list

# predict points from pointmap

# def predict_pointcloud_from_images(image_paths: List[str], model, device: str, 
#                                  pixel_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
#     """
#     从图像预测点云并获取指定像素的3D坐标
    
#     Args:
#         image_paths: 图像路径列表
#         model: VGGT模型  
#         device: 设备
#         pixel_coords: 像素坐标 (N, 2)
        
#     Returns:
#         pred_points: 预测点云 (M, 3)
#         pred_colors: 点云颜色 (M, 3)  
#         specific_points_list: 每张图像对应像素的3D坐标列表
#     """
#     if len(image_paths) == 0:
#         raise ValueError("图像路径列表为空")
    
#     print(f"处理 {len(image_paths)} 张图像进行点云预测")
    
#     # 加载和预处理图像 - 使用多坐标版本
#     images, coords_list = load_and_preprocess_images(image_paths, pixel_coords=pixel_coords)
#     images = images.to(device)
    
#     # 将坐标移到设备上
#     for i in range(len(coords_list)):
#         if coords_list[i] is not None:
#             coords_list[i] = coords_list[i].to(device)
    
#     print(f"预处理后的图像形状: {images.shape}")
    
#     # 根据GPU能力设置精度
#     if torch.cuda.is_available():
#         dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
#     else:
#         dtype = torch.float32
    
#     # 运行推理
#     print("运行推理...")
#     with torch.no_grad():
#         if torch.cuda.is_available():
#             with torch.amp.autocast('cuda', dtype=dtype):
#                 predictions = model(images)
#         else:
#             predictions = model(images)
    
#     print(f"预测结果键: {list(predictions.keys())}")
    
#     # 直接使用模型预测的world_points，无需重新计算
#     world_points = predictions["world_points"]  # Shape: [1, 4, 518, 518, 3]
#     world_points_conf = predictions.get("world_points_conf", None)  # 置信度（如果有的话）
    
#     # 移除批次维度
#     if world_points.dim() == 5 and world_points.shape[0] == 1:
#         world_points = world_points.squeeze(0)  # Shape: [4, 518, 518, 3]
    
#     if world_points_conf is not None:
#         if world_points_conf.dim() == 5 and world_points_conf.shape[0] == 1:
#             world_points_conf = world_points_conf.squeeze(0)  # Shape: [4, 518, 518]
    
#     print(f"World points shape: {world_points.shape}")
#     if world_points_conf is not None:
#         print(f"World points confidence shape: {world_points_conf.shape}")
    
#     # 提取点云和颜色
#     print("提取点云颜色...")
#     all_points = []
#     all_colors = []
#     specific_points_list = []
    
#     # 获取图像维度
#     num_frames, H, W, _ = world_points.shape
    
#     # 为特定像素坐标创建结果列表
#     for frame_idx in range(len(image_paths)):
#         specific_points_list.append(torch.full((pixel_coords.shape[0], 3), float('nan'), device=device))
    
#     for i, image_path in enumerate(image_paths):
#         # 跳过超出范围的索引
#         if i >= num_frames:
#             print(f"跳过图像 {i}，超出world_points范围")
#             continue
            
#         # 获取当前帧的点云 - 直接从预测结果获取
#         frame_points = world_points[i]  # Shape: [518, 518, 3]
        
#         # 加载原始图像以获取颜色
#         try:
#             # 加载图像
#             img = Image.open(image_path)
#             img_np = np.array(img)
            
#             # 确保图像是RGB
#             if len(img_np.shape) == 2:  # 灰度图
#                 img_np = np.stack([img_np, img_np, img_np], axis=2)
#             elif img_np.shape[2] == 4:  # RGBA
#                 img_np = img_np[:, :, :3]
                
#             # 转换为PyTorch张量
#             img_tensor = torch.from_numpy(img_np).float().to(device) / 255.0
            
#             # 调整图像大小以匹配点云维度 (518x518)
#             if img_tensor.shape[0] != H or img_tensor.shape[1] != W:
#                 img_tensor = torch.nn.functional.interpolate(
#                     img_tensor.permute(2, 0, 1).unsqueeze(0),  # [1, C, H, W]
#                     size=(H, W),
#                     mode='bilinear',
#                     align_corners=False
#                 ).squeeze(0).permute(1, 2, 0)  # [H, W, C]
            
#             # 直接从point map获取特定像素坐标对应的世界坐标
#             frame_coords = coords_list[i] if i < len(coords_list) else None
#             if frame_coords is not None:
#                 for j in range(frame_coords.shape[0]):
#                     if j < pixel_coords.shape[0]:  # 确保索引不越界
#                         y, x = frame_coords[j].long()
#                         # 检查坐标是否在范围内
#                         if 0 <= y < H and 0 <= x < W:
#                             specific_points_list[i][j] = frame_points[y, x]
            
#             # 展平点云和颜色
#             flat_points = frame_points.reshape(-1, 3)  # [518*518, 3]
#             flat_colors = img_tensor.reshape(-1, 3)    # [518*518, 3]
            
#             # 创建掩码数组，标记需要保留的点
#             mask = torch.ones(flat_points.shape[0], dtype=torch.bool, device=device)
            
#             # 基本过滤：移除无效点
#             finite_mask = torch.isfinite(flat_points).all(dim=1)
#             valid_norm_mask = torch.norm(flat_points, dim=1) > 1e-6
#             mask = mask & finite_mask & valid_norm_mask
            
#             # 应用世界坐标点置信度阈值
#             if world_points_conf is not None and i < world_points_conf.shape[0]:
#                 conf_frame = world_points_conf[i]  # Shape: [518, 518]
#                 conf_flat = conf_frame.reshape(-1)  # [518*518]
                
#                 # 确保维度匹配
#                 if conf_flat.shape[0] == mask.shape[0]:
#                     conf_mask = conf_flat > 0.8  # 根据需要调整阈值
#                     mask = mask & conf_mask
#                     print(f"图像 {i}: 应用置信度过滤，阈值=0.5")
            
#             # # 可选：应用深度置信度阈值
#             # if "depth_conf" in predictions:
#             #     depth_conf = predictions["depth_conf"]
#             #     if isinstance(depth_conf, torch.Tensor):
#             #         # 移除批次维度
#             #         if depth_conf.dim() == 4 and depth_conf.shape[0] == 1:
#             #             depth_conf = depth_conf.squeeze(0)  # [4, 518, 518]
                    
#             #         if i < depth_conf.shape[0]:
#             #             depth_conf_frame = depth_conf[i]
#             #             depth_conf_flat = depth_conf_frame.reshape(-1)
                        
#             #             # 确保维度匹配
#             #             if depth_conf_flat.shape[0] == mask.shape[0]:
#             #                 depth_conf_mask = depth_conf_flat > 0.7  # 根据需要调整阈值
#             #                 mask = mask & depth_conf_mask
#             #                 print(f"图像 {i}: 应用深度置信度过滤，阈值=0.7")
            
#             # 应用掩码
#             flat_points = flat_points[mask]
#             flat_colors = flat_colors[mask]
            
#             if flat_points.shape[0] > 0:
#                 all_points.append(flat_points)
#                 all_colors.append(flat_colors)
#                 print(f"图像 {i}: 添加了 {flat_points.shape[0]} 个有效点")
#             else:
#                 print(f"图像 {i}: 没有有效点")
            
#         except Exception as e:
#             print(f"处理图像 {image_path} 时出错: {e}")
#             continue
    
#     # 如果没有有效点，返回空张量
#     if len(all_points) == 0:
#         print("警告: 没有生成任何有效点")
#         return torch.zeros((0, 3)), torch.zeros((0, 3)), specific_points_list
    
#     # 组合所有点和颜色
#     points = torch.cat(all_points, dim=0)
#     colors = torch.cat(all_colors, dim=0)
    
#     print(f"生成的点云有 {points.shape[0]} 个点")
    
#     return points, colors, specific_points_list

def generate_point_cloud_from_folders_vggt(folder1: str, folder2: str, model: Optional[VGGT] = None,
                                           device: str = None, 
                                          filter_sky: bool = False, filter_black_bg: bool = False, 
                                          filter_white_bg: bool = False, pixel_coords = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    从两个文件夹中的图像生成彩色点云
    
    Args:
        folder1: 包含PNG图像的第一个文件夹的路径
        folder2: 包含PNG图像的第二个文件夹的路径
        model: 预加载的VGGT模型（可选，如果未提供将加载一个）
        device: 设备（'cuda'或'cpu'）。如果为None，将使用可用的CUDA
        filter_sky: 是否过滤天空区域（默认为False）
        filter_black_bg: 是否过滤黑色背景（默认为False）
        filter_white_bg: 是否过滤白色背景（默认为False）
        pixel_coords: 形状为(n, 2)的张量，包含n个像素点的坐标[y, x]
        
    Returns:
        points: 形状为(N, 3)的张量，包含3D点坐标
        colors: 形状为(N, 3)的张量，包含每个点的RGB颜色
        extrinsic: 外参矩阵
        specific_world_points: 形状为(n, 3)的张量，包含指定像素点对应的3D世界坐标
    """
    
    # 分别收集两个文件夹中的PNG图像
    folder1_images = glob.glob(os.path.join(folder1, "*.png"))
    folder2_images = glob.glob(os.path.join(folder2, "*.png"))
    def natural_sort_key(s):
        import re
        # 提取文件名中的数字部分用于排序
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split(r'(\d+)', os.path.basename(s))]
    
    folder1_images = sorted(folder1_images, key=natural_sort_key)
    folder2_images = sorted(folder2_images)
    image_paths = folder1_images + folder2_images
    
    if len(image_paths) == 0:
        raise ValueError(f"在文件夹 {folder1} 和 {folder2} 中未找到PNG图像")
    
    print(f"找到 {len(image_paths)} 张图像")
    
    # 加载和预处理图像
    images, new_pixel_coords = load_and_preprocess_images(image_paths, pixel_coords=pixel_coords)
    images = images.to(device)
    if new_pixel_coords is not None:
        new_pixel_coords = new_pixel_coords.to(device)
    print(f"预处理后的图像形状: {images.shape}")
    

    
    # 根据GPU能力设置精度
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    
    # 运行推理
    print("运行推理...")
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)
    
    # 将姿态编码转换为外部和内部矩阵
    print("将姿态编码转换为相机参数...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # 获取深度图
    print("从深度图计算世界点...")
    depth_map = predictions["depth"]  # 形状: [1, 21, 252, 518, 1]
    
    # 移除额外的批次维度
    if depth_map.dim() == 5 and depth_map.shape[0] == 1:
        depth_map = depth_map.squeeze(0)  # 形状: [21, 252, 518, 1]
    
    if extrinsic.dim() == 4 and extrinsic.shape[0] == 1:
        extrinsic = extrinsic.squeeze(0)  # 移除额外的批次维度
    
    if intrinsic.dim() == 4 and intrinsic.shape[0] == 1:
        intrinsic = intrinsic.squeeze(0)  # 移除额外的批次维度
    
    # 手动实现unproject_depth_map_to_point_map的逻辑
    world_points_list = []
    num_frames = depth_map.shape[0]
    
    # 转换为numpy数组
    depth_map_np = depth_map.detach().cpu().numpy()
    extrinsic_np = extrinsic.detach().cpu().numpy()
    intrinsic_np = intrinsic.detach().cpu().numpy()
    
    # 为特定像素坐标创建结果变量
    specific_world_points = None
    
    # 对每一帧处理
    for frame_idx in range(num_frames):
        # 获取当前帧的深度图，确保它是2D的
        cur_depth = depth_map_np[frame_idx]
        if cur_depth.ndim > 2:
            cur_depth = cur_depth.squeeze(-1)  # 移除通道维度，变成[H, W]
        
        # 获取当前帧的相机参数
        cur_extrinsic = extrinsic_np[frame_idx]
        cur_intrinsic = intrinsic_np[frame_idx]
        
        # 调用depth_to_world_coords_points函数
        try:
            cur_world_points, _, _ = depth_to_world_coords_points(
                cur_depth, cur_extrinsic, cur_intrinsic
            )
            world_points_list.append(cur_world_points)
        except Exception as e:
            print(f"处理帧 {frame_idx} 时出错: {e}")
            continue
    
    # 如果没有成功处理任何帧，返回空张量
    if len(world_points_list) == 0:
        return torch.zeros((0, 3), device=device), torch.zeros((0, 3), device=device), extrinsic, None
    
    # 将所有帧的世界坐标点堆叠起来
    world_points_np = np.stack(world_points_list, axis=0)
    world_points = torch.from_numpy(world_points_np).to(device)
    
    # 从原始图像中提取每个点的颜色
    print("提取点云颜色...")
    all_points = []
    all_colors = []
    
    # 导入PIL和numpy用于图像处理
    from PIL import Image
    
    # 为特定像素坐标创建结果 - 初始化为NaN
    if new_pixel_coords is not None:
        specific_world_points = torch.full((new_pixel_coords.shape[0], 3), float('nan'), device=device)
    
    for i, image_path in enumerate(image_paths):
        # 跳过超出范围的索引
        if i >= world_points.shape[0]:
            continue
            
        # 获取当前帧的点云
        frame_points = world_points[i]  # (H, W, 3)
        H, W, _ = frame_points.shape
        
        # 加载原始图像以获取颜色
        try:
            # 加载图像
            img = Image.open(image_path)
            img_np = np.array(img)
            
            # 确保图像是RGB
            if len(img_np.shape) == 2:  # 灰度图
                img_np = np.stack([img_np, img_np, img_np], axis=2)
            elif img_np.shape[2] == 4:  # RGBA
                img_np = img_np[:, :, :3]
                
            # 转换为PyTorch张量
            img_tensor = torch.from_numpy(img_np).float().to(device) / 255.0
            
            # 调整图像大小以匹配点云
            if img_tensor.shape[0] != H or img_tensor.shape[1] != W:
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor.permute(2, 0, 1).unsqueeze(0),  # [1, C, H, W]
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0)  # [H, W, C]
            
            # 为最后一帧获取特定像素坐标对应的点云和掩码状态
            if i == len(folder1_images) - 1 and new_pixel_coords is not None:
                # 创建一个2D有效点掩码，用于记录哪些像素点是有效的
                valid_mask_2d = torch.ones((H, W), dtype=torch.bool, device=device)
                
                # 应用过滤条件前获取当前索引下的世界坐标
                # 注意：这是直接获取3D坐标，而不考虑后续过滤
                for j in range(new_pixel_coords.shape[0]):
                    y, x = new_pixel_coords[j].long()
                    # 检查坐标是否在范围内
                    if 0 <= y < H and 0 <= x < W:
                        specific_world_points[j] = torch.tensor(frame_points[y, x], device=device)
            
            # 展平点云和颜色
            flat_points = frame_points.reshape(-1, 3)
            flat_colors = img_tensor.reshape(-1, 3)
            
            # 创建掩码数组，标记需要保留的点
            mask = torch.ones(flat_points.shape[0], dtype=torch.bool, device=device)
            
            # 过滤黑色背景
            if filter_black_bg:
                # 计算RGB平均值，低于阈值的被视为黑色背景
                rgb_mean = flat_colors.mean(dim=1)
                black_mask = rgb_mean > 0.1  # 阈值可调整
                mask = mask & black_mask
            
            # 过滤白色背景
            if filter_white_bg:
                # 计算RGB平均值，高于阈值的被视为白色背景
                rgb_mean = flat_colors.mean(dim=1)
                white_mask = rgb_mean < 0.9  # 阈值可调整
                mask = mask & white_mask
            
            # 应用置信度阈值（如果可用）
            if "depth_conf" in predictions:
                confidence = predictions["depth_conf"]
                if isinstance(confidence, torch.Tensor):
                    # 确保有正确的形状
                    if confidence.dim() == 4 and confidence.shape[0] == 1:
                        confidence = confidence.squeeze(0)  # 移除批次维度
                    
                    if i < confidence.shape[0]:
                        conf_frame = confidence[i]
                        conf_flat = conf_frame.reshape(-1)
                        # 确保维度匹配
                        if conf_flat.shape[0] == mask.shape[0]:
                            conf_mask = conf_flat > 0.7  # 根据需要调整阈值
                            mask = mask & conf_mask
            
            # 对于最后一帧，更新特定像素点的世界坐标有效性
            if i == len(folder1_images) - 1 and new_pixel_coords is not None:
                # 将1D掩码重塑为2D掩码
                mask_2d = mask.reshape(H, W)
                
                # 检查特定像素点是否被过滤掉了
                for j in range(new_pixel_coords.shape[0]):
                    y, x = new_pixel_coords[j].long()
                    # 检查坐标是否在范围内且掩码为True
                    if 0 <= y < H and 0 <= x < W and not mask_2d[y, x]:
                        # 如果该点被过滤掉，将世界坐标设为NaN
                        specific_world_points[j] = torch.tensor([float('nan'), float('nan'), float('nan')], device=device)
            
            # 应用掩码
            flat_points = flat_points[mask]
            flat_colors = flat_colors[mask]
            
            all_points.append(flat_points)
            all_colors.append(flat_colors)
            
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            continue
    
    # 如果没有有效点，返回空张量
    if len(all_points) == 0:
        return torch.zeros((0, 3), device=device), torch.zeros((0, 3), device=device), extrinsic, specific_world_points
    
    # 组合所有点和颜色
    points = torch.cat(all_points, dim=0)
    colors = torch.cat(all_colors, dim=0)
    
    print(f"生成的点云有 {points.shape[0]} 个点")
    
    return points, colors, extrinsic, specific_world_points

def save_point_cloud_as_ply(filename: str, points: torch.Tensor, colors: torch.Tensor):
    """
    将点云及其颜色保存为PLY文件
    
    Args:
        filename: 输出文件名 (.ply)
        points: 形状为(N, 3)的张量，包含3D点坐标
        colors: 形状为(N, 3)的张量，包含RGB颜色（值在[0, 1]范围内）
    """
    # 确保点和颜色在CPU上
    points_cpu = points.detach().cpu()
    colors_cpu = colors.detach().cpu()
    
    # 确保颜色在正确的范围[0, 255]内以适用于PLY格式
    colors_uint8 = (colors_cpu * 255).to(torch.uint8)
    
    # 写入PLY文件头
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points_cpu.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 写入点数据
        for i in range(points_cpu.shape[0]):
            f.write(f"{points_cpu[i, 0].item()} {points_cpu[i, 1].item()} {points_cpu[i, 2].item()} "
                    f"{colors_uint8[i, 0].item()} {colors_uint8[i, 1].item()} {colors_uint8[i, 2].item()}\n")
    
    print(f"点云已保存至 {filename}")

# # 示例用法:
# if __name__ == "__main__":
    
#     # 加载模型一次以重复使用
#     model = load_vggt_model("cuda:0")
    
#     # # 生成点云
#     # points, colors = generate_point_cloud_from_folders(folder1, folder2, model)
    












##########################################################################
######## The following code is for reconstruction: CUT3R ################
##########################################################################


def load_reconstruction_model(model_path="/home/sli/phd_projects/imagination-outdoor/CUT3R/src/cut3r_512_dpt_4_64.pth", device="cuda:0"):
    """
    加载ARCroco3DStereo模型

    Args:
        model_path (str): 模型权重文件的路径
        device (str): 计算设备，"cuda"或"cpu"

    Returns:
        model: 加载的模型
        state_dict: 模型状态字典
    """

    # 添加模型路径到环境变量（如果需要）
    # 这取决于您的环境配置，可能需要修改
    model_dir = os.path.dirname(model_path)
    if model_dir not in sys.path:
        sys.path.append(model_dir)

    # 导入模型类
    from CUT3R.src.dust3r.model import ARCroco3DStereo

    # 加载模型
    model = ARCroco3DStereo.from_pretrained(model_path).to(device)
    model.eval() 
    print("reconstruction model loaded!")

    state_args = {}
    
    return model, state_args





def prepare_input_for_reconstruction(img_paths, img_mask, size, raymaps=None, raymap_mask=None, revisit=1, update=True):
    """
    准备推理的输入视图
    
    Args:
        img_paths (list): 图像路径列表
        img_mask (list): 图像有效性掩码
        size (int): 目标图像大小
        raymaps (list, optional): 射线映射列表
        raymap_mask (list, optional): 射线映射有效性掩码
        revisit (int): 每个视图重访次数
        update (bool): 重访时是否更新状态
        
    Returns:
        list: 视图字典列表
    """
    # 导入图像加载器
    from CUT3R.src.dust3r.utils.image import load_images
    
    # 加载图像
    images = load_images(img_paths, size=size)
    views = []
    
    if raymaps is None and raymap_mask is None:
        # 只有图像的情况
        for i in range(len(images)):
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            views.append(view)
    else:
        # 图像和射线映射混合的情况
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)
        
        j = 0
        k = 0
        for i in range(num_views):
            view = {
                "img": (
                    images[j]["img"]
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)
                ),
                "ray_map": (
                    raymaps[k]
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)
                ),
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
        assert j == len(images) and k == len(raymaps)
    
    # 处理重访逻辑
    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views
    
    return views



def process_new_images_for_reconstruction(model, image_paths, state_args=None, indices=None, size=512, 
                                          device="cuda", save_state=False, state_path=None, pose_i=None):
    """
    处理新图像并更新状态，提取GT姿态
    
    Args:
        model: ARCroco3DStereo模型
        image_paths: 图像路径列表或包含图像的文件夹路径
        state_args: 旧状态参数，如果为None则从头开始
        indices: 要处理的图像索引，如果为None则处理所有图像
        size: 图像调整目标大小
        device: 计算设备
        save_state: 是否保存状态
        state_path: 状态保存路径
        pose_i: 如果提供，则选择(pose_i+1)*4到(pose_i+1)*4-3的PT文件
        
    Returns:
        tuple: (outputs, updated_state_args, gt_poses)
    """
    # 导入必要函数
    from CUT3R.src.dust3r.inference import inference, inference_recurrent
    from CUT3R.src.dust3r.utils.image import load_images
    import os
    import glob
    import torch
    import numpy as np
    from copy import deepcopy
    
    # 选择要处理的图像
    selected_images = []
    gt_poses = []
    
    if pose_i is not None:
        # 处理PT文件的情况
        if isinstance(image_paths, str) and os.path.isdir(image_paths):
            pt_files = sorted(glob.glob(os.path.join(image_paths, "*.pt")))
            indices_to_select = [
                (pose_i + 1) * 4,
                (pose_i + 1) * 4 - 1,
                (pose_i + 1) * 4 - 2,
                (pose_i + 1) * 4 - 3
            ]
            valid_indices = [i for i in indices_to_select if i < len(pt_files)]
            selected_images = [pt_files[i] for i in valid_indices]
        else:
            indices_to_select = [
                (pose_i + 1) * 4,
                (pose_i + 1) * 4 - 1,
                (pose_i + 1) * 4 - 2,
                (pose_i + 1) * 4 - 3
            ]
            selected_images = [image_paths[i] for i in indices_to_select if i < len(image_paths)]
    else:
        # 处理普通图像文件的情况
        if isinstance(image_paths, str) and os.path.isdir(image_paths):
            selected_images = sorted(glob.glob(os.path.join(image_paths, "*.png")))
        else:
            if indices is not None:
                selected_images = [image_paths[i] for i in indices if i < len(image_paths)]
            else:
                selected_images = image_paths
    
    if not selected_images:
        raise ValueError("没有找到可处理的图像文件")
    
    img_mask = [True] * len(selected_images)
    
    # PT文件处理 - 提取图像和GT姿态
    if pose_i is not None:
        import tempfile
        import imageio.v2 as iio
        
        temp_dir = tempfile.mkdtemp()
        temp_images = []
        
        try:
            for i, pt_path in enumerate(selected_images):
                camera_data = torch.load(pt_path)
                img_tensor = camera_data['rgb'].squeeze()
                
                # 提取GT姿态
                if 'R' in camera_data and 'T' in camera_data:
                    R = camera_data['R']
                    if isinstance(R, torch.Tensor):
                        R = R.clone().detach()
                    else:
                        R = torch.tensor(R, dtype=torch.float32)
                        
                    T = camera_data['T']
                    if isinstance(T, torch.Tensor):
                        T = T.clone().detach()
                    else:
                        T = torch.tensor(T, dtype=torch.float32)
                    
                    # 处理批次维度
                    if len(R.shape) == 3:
                        R = R.squeeze(0)  # [1, 3, 3] -> [3, 3]
                    if len(T.shape) == 2:
                        T = T.squeeze(0)  # [1, 3] -> [3]
                    
                    # 构建4x4转换矩阵 - 使用CPU，后续再移至需要的设备
                    pose_matrix = torch.eye(4, dtype=torch.float32)
                    pose_matrix[:3, :3] = R
                    pose_matrix[:3, 3] = T
                    gt_poses.append(pose_matrix)
                
                # 处理图像数据
                if img_tensor.shape[0] == 3:  # 如果是[3, H, W]格式
                    img_tensor = img_tensor.permute(1, 2, 0)
                
                if img_tensor.max() <= 1.0:
                    img_tensor = img_tensor * 255.0
                
                img_np = img_tensor.cpu().numpy().astype(np.uint8)
                temp_img_path = os.path.join(temp_dir, f"temp_img_{i}.png")
                iio.imwrite(temp_img_path, img_np)
                temp_images.append(temp_img_path)
            
            # 准备视图
            views = prepare_input_for_reconstruction(
                img_paths=temp_images,
                img_mask=img_mask,
                size=size,
                revisit=1,
                update=True
            )
        finally:
            # 清理临时文件
            import shutil
            shutil.rmtree(temp_dir)
    else:
        # 普通图像处理
        views = prepare_input_for_reconstruction(
            img_paths=selected_images,
            img_mask=img_mask,
            size=size,
            revisit=1,
            update=True
        )
    
    # 执行推理
    if state_args is not None and state_args:
        outputs, updated_state_args = inference_recurrent(views, model, device, state_args)
    else:
        outputs, updated_state_args = inference(views, model, device)
    
    # 保存状态
    if save_state and state_path:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        torch.save(updated_state_args, state_path)
    
    # 返回结果
    return outputs, updated_state_args, gt_poses


def extract_point_cloud(outputs, existing_point_cloud=None, output_dir=None, vis_threshold=1.5, use_pose=True, to_gpu=True, format='pt', gt_poses=None):
    """
    从模型输出中提取点云数据，可选择使用GT姿态
    
    Args:
        outputs: 模型推理输出
        existing_point_cloud: 现有点云数据
        output_dir: 输出目录
        vis_threshold: 可视化阈值
        use_pose: 是否使用相机姿态
        to_gpu: 是否将数据保存到GPU
        format: 保存格式 ('pt' 或 'npy')
        gt_poses: 真实相机姿态列表
        
    Returns:
        dict: 点云数据字典
    """
    import torch
    import numpy as np
    import os
    from CUT3R.src.dust3r.utils.camera import pose_encoding_to_camera
    from CUT3R.src.dust3r.post_process import estimate_focal_knowing_depth
    from CUT3R.src.dust3r.utils.geometry import geotrf
    
    # 处理输出
    pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
    
    # 使用相机姿态变换点
    if use_pose:
        # 判断是否有GT姿态可用
        if gt_poses is not None and len(gt_poses) > 0:
            # 确保有足够的GT姿态
            if len(gt_poses) < len(outputs["pred"]):
                # 复制最后一个姿态来补充
                last_pose = gt_poses[-1]
                while len(gt_poses) < len(outputs["pred"]):
                    gt_poses.append(last_pose.clone())
            pr_poses = gt_poses
        else:
            # 使用模型预测的姿态
            pr_poses = [
                pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
                for pred in outputs["pred"]
            ]
        
        # 应用相机姿态变换
        pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self_ls):
            # 确保设备匹配
            if isinstance(pose, torch.Tensor) and pose.device != pself.device:
                pose = pose.to(pself.device)
            
            # 处理批次维度
            if len(pose.shape) == 2:  # [4, 4]
                pose_batch = pose.unsqueeze(0)
            else:  # [B, 4, 4]
                pose_batch = pose
            
            # 变换点云
            pts3ds_other.append(geotrf(pose_batch, pself.unsqueeze(0)))
        
        conf_other = conf_self
    else:
        # 不使用姿态变换，直接使用模型输出的点云
        pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
        conf_other = [output["conf"].cpu() for output in outputs["pred"]]
    
    # 为相机参数计算焦距
    sample_pts = pts3ds_self_ls[0].to('cpu')  # 使用第一个视图的点云计算，移到CPU避免设备不匹配
    B, *spatial_dims, _ = sample_pts.shape
    if len(spatial_dims) == 3:  # [B, 1, H, W, 3]形状
        H, W = spatial_dims[1], spatial_dims[2]
    else:  # [B, H, W, 3]形状
        H, W = spatial_dims[0], spatial_dims[1]
        
    pp = torch.tensor([W // 2, H // 2], device=sample_pts.device).float().repeat(B, 1)
    # 调整点云形状以适合estimate_focal_knowing_depth函数
    if len(spatial_dims) == 3:
        pts_for_focal = sample_pts.squeeze(1)  # 移除多余维度
    else:
        pts_for_focal = sample_pts
        
    focal = estimate_focal_knowing_depth(pts_for_focal, pp, focal_mode="weiszfeld")
    
    # 提取颜色
    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]
    
    # 恢复相机姿态的R和t部分
    if use_pose:
        R_c2w_list = []
        t_c2w_list = []
        
        for pose in pr_poses:
            # 确保姿态在CPU上进行后处理
            if isinstance(pose, torch.Tensor) and pose.device.type != 'cpu':
                pose = pose.cpu()
                
            if len(pose.shape) == 2:  # [4, 4]形状
                R_c2w_list.append(pose[:3, :3].unsqueeze(0))
                t_c2w_list.append(pose[:3, 3].unsqueeze(0))
            else:  # 已经有批次维度 [B, 4, 4]
                R_c2w_list.append(pose[:, :3, :3])
                t_c2w_list.append(pose[:, :3, 3])
                
        R_c2w = torch.cat(R_c2w_list, 0)
        t_c2w = torch.cat(t_c2w_list, 0)
    else:
        # 如果不使用姿态，创建单位变换
        R_c2w = torch.cat([torch.eye(3).unsqueeze(0) for _ in range(len(pts3ds_other))], 0)
        t_c2w = torch.cat([torch.zeros(1, 3) for _ in range(len(pts3ds_other))], 0)
    
    # 准备相机参数
    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
        "R": R_c2w.cpu().numpy(),
        "t": t_c2w.cpu().numpy(),
    }
    
    # 根据置信度过滤点云
    filtered_pts = []
    filtered_colors = []
    
    for pts, conf, col in zip(pts3ds_other, conf_other, colors):
        # 处理形状不一致
        if len(pts.shape) > 3:  # 如果有额外的维度
            pts = pts.squeeze(1)  # 移除多余的批次维度
        
        # 确保conf形状匹配
        if len(conf.shape) < len(pts.shape) - 1:
            conf = conf.view(*pts.shape[:-1])
        
        # 过滤点云
        mask = conf > vis_threshold
        flat_pts = pts.reshape(-1, 3)
        flat_mask = mask.reshape(-1)
        
        # 处理大小不匹配
        if flat_mask.shape[0] != flat_pts.shape[0]:
            if flat_mask.shape[0] > flat_pts.shape[0]:
                flat_mask = flat_mask[:flat_pts.shape[0]]
            else:
                temp_mask = torch.zeros(flat_pts.shape[0], dtype=torch.bool, device=flat_mask.device)
                temp_mask[:flat_mask.shape[0]] = flat_mask
                flat_mask = temp_mask
        
        # 处理颜色形状
        if len(col.shape) == 4:  # [B, H, W, C]
            flat_col = col.reshape(-1, col.shape[-1])
        else:  # [B, C, H, W]
            flat_col = col.permute(0, 2, 3, 1).reshape(-1, col.shape[1])
        
        # 确保颜色数量匹配
        if flat_col.shape[0] != flat_pts.shape[0]:
            if flat_col.shape[0] < flat_pts.shape[0]:
                last_color = flat_col[-1].unsqueeze(0)
                extra_colors = last_color.repeat(flat_pts.shape[0] - flat_col.shape[0], 1)
                flat_col = torch.cat([flat_col, extra_colors], dim=0)
            else:
                flat_col = flat_col[:flat_pts.shape[0]]
        
        # 应用过滤
        filtered_pts.append(flat_pts[flat_mask])
        filtered_colors.append(flat_col[flat_mask])
    
    # 创建点云数据结构
    point_cloud_data = {
        'pts3ds': filtered_pts,
        'colors': filtered_colors,
        'conf': conf_other,
        'cam_dict': cam_dict
    }
    
    # 合并现有点云（如果有）
    if existing_point_cloud is not None:
        point_cloud_data = merge_point_clouds(point_cloud_data, existing_point_cloud)
    
    # 保存点云数据（如果需要）
    if output_dir:
        save_point_cloud_data(point_cloud_data, output_dir, format=format)
    
    # 转换到GPU（如果需要）
    if to_gpu:
        point_cloud_data = to_gpu_data(point_cloud_data)
    
    return point_cloud_data


def merge_point_clouds(new_point_cloud, existing_point_cloud):
    """
    合并两个点云数据集（兼容PyTorch张量和NumPy数组）
    
    Args:
        new_point_cloud: 新点云数据（可能包含张量或NumPy数组）
        existing_point_cloud: 现有点云数据（可能包含张量或NumPy数组）
        
    Returns:
        dict: 合并后的点云数据（统一为NumPy数组）
    """
    import numpy as np
    import torch

    def _to_numpy(x):
        """将输入转换为NumPy数组（兼容张量和数组）"""
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.array(x)  # 确保即使是列表也转为NumPy数组

    # 合并点、颜色和置信度（假设这些是列表）
    merged_pts = existing_point_cloud['pts3ds'] + new_point_cloud['pts3ds']
    merged_colors = existing_point_cloud['colors'] + new_point_cloud['colors']
    merged_conf = existing_point_cloud['conf'] + new_point_cloud['conf']

    # 合并相机参数（处理混合类型）
    e_cam = existing_point_cloud['cam_dict']
    n_cam = new_point_cloud['cam_dict']

    merged_cam_dict = {
        "focal": np.concatenate([_to_numpy(e_cam["focal"]), _to_numpy(n_cam["focal"])]),
        "pp": np.concatenate([_to_numpy(e_cam["pp"]), _to_numpy(n_cam["pp"])]),
        "R": np.concatenate([_to_numpy(e_cam["R"]), _to_numpy(n_cam["R"])]),
        "t": np.concatenate([_to_numpy(e_cam["t"]), _to_numpy(n_cam["t"])]),
    }

    return {
        'pts3ds': merged_pts,
        'colors': merged_colors,
        'conf': merged_conf,
        'cam_dict': merged_cam_dict
    }

def to_gpu_data(point_cloud_data):
    """
    将点云数据转移到GPU
    
    Args:
        point_cloud_data: 点云数据
        
    Returns:
        dict: GPU上的点云数据
    """
    import torch
    import numpy as np
    
    # 转换点云和颜色
    gpu_pts = [p.cuda() if isinstance(p, torch.Tensor) and p.device.type == 'cpu' else p for p in point_cloud_data['pts3ds']]
    gpu_colors = [c.cuda() if isinstance(c, torch.Tensor) and c.device.type == 'cpu' else c for c in point_cloud_data['colors']]
    gpu_conf = [c.cuda() if isinstance(c, torch.Tensor) and c.device.type == 'cpu' else c for c in point_cloud_data['conf']]
    
    # 转换相机参数
    gpu_cam_dict = {}
    for k, v in point_cloud_data['cam_dict'].items():
        if isinstance(v, np.ndarray):
            gpu_cam_dict[k] = torch.from_numpy(v).cuda()
        elif isinstance(v, torch.Tensor) and v.device.type == 'cpu':
            gpu_cam_dict[k] = v.cuda()
        else:
            gpu_cam_dict[k] = v
    
    return {
        'pts3ds': gpu_pts,
        'colors': gpu_colors,
        'conf': gpu_conf,
        'cam_dict': gpu_cam_dict
    }


def save_point_cloud_data(point_cloud_data, output_dir, format='pt'):
    """
    保存点云数据到指定目录
    
    Args:
        point_cloud_data: 点云数据
        output_dir: 输出目录
        format: 保存格式 ('pt' 或 'npy')
    """
    import os
    import torch
    import numpy as np
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    if format == 'pt':
        # 保存为PyTorch格式
        torch.save(point_cloud_data, os.path.join(output_dir, 'point_cloud_data.pt'))
    else:
        # 保存为NumPy格式
        os.makedirs(os.path.join(output_dir, "points"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "colors"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "conf"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "camera"), exist_ok=True)
        
        # 保存点云数据
        for i, (pts, col, conf) in enumerate(zip(
            point_cloud_data['pts3ds'], 
            point_cloud_data['colors'], 
            point_cloud_data['conf']
        )):
            np.save(os.path.join(output_dir, "points", f"{i:06d}.npy"), pts.cpu().numpy())
            np.save(os.path.join(output_dir, "colors", f"{i:06d}.npy"), col.cpu().numpy())
            np.save(os.path.join(output_dir, "conf", f"{i:06d}.npy"), conf.cpu().numpy())
        
        # 保存相机参数
        np.savez(
            os.path.join(output_dir, "camera", "camera_params.npz"),
            focal=point_cloud_data['cam_dict']['focal'],
            pp=point_cloud_data['cam_dict']['pp'],
            R=point_cloud_data['cam_dict']['R'],
            t=point_cloud_data['cam_dict']['t']
        )


def extract_points_and_colors(point_cloud_data, device):
    """
    从点云数据中提取点坐标和颜色（增强空张量处理）
    
    Args:
        point_cloud_data: 点云数据字典
        device: 计算设备
        
    Returns:
        tuple: (points, colors) 其中:
            - points: 形状为(n, 3)的张量，包含所有点坐标
            - colors: 形状为(n, 3)的张量，包含所有点颜色
    """
    import torch
    
    # 合并所有点云和颜色
    all_points = []
    all_colors = []
    
    for pts, colors in zip(point_cloud_data['pts3ds'], point_cloud_data['colors']):
        # 跳过空数据
        if len(pts) == 0 or len(colors) == 0:
            continue
            
        # 确保数据是张量
        if not isinstance(pts, torch.Tensor):
            pts = torch.tensor(pts, dtype=torch.float32, device=device)
        
        if not isinstance(colors, torch.Tensor):
            colors = torch.tensor(colors, dtype=torch.float32, device=device)
            
        # 重新整形以确保形状正确
        if len(pts.shape) > 2:
            pts = pts.reshape(-1, 3)
            
        if len(colors.shape) > 2:
            colors = colors.reshape(-1, 3)
            
        # 安全检查颜色值范围（处理空张量情况）
        if colors.numel() > 0:  # 只在张量非空时检查
            if colors.max() > 1.0:
                colors = colors / 255.0
            
        all_points.append(pts)
        all_colors.append(colors)
    
    # 合并结果
    if len(all_points) > 0:
        points = torch.cat(all_points, dim=0)
        colors = torch.cat(all_colors, dim=0)
    else:
        points = torch.zeros((0, 3), dtype=torch.float32, device=device)
        colors = torch.zeros((0, 3), dtype=torch.float32, device=device)
    
    return points, colors
