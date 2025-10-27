import copy
import json
import os.path as osp
import torch
import os
import numpy as np
import glob
import splines
import splines.quaternion
from copy import deepcopy
from datetime import datetime
import torch
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator
import numpy as np
from einops import rearrange
import sys
from pathlib import Path
import copy
import os
import torch.nn.functional as F
from einops import rearrange
import imageio.v3 as iio
from typing import List, Dict, Tuple, Any, Optional, Literal
project_root = Path(__file__).parent.parent  # 三级父目录：imagination-outdoor/
sys.path.append(str(project_root))

import logging

# 将全局日志等级设置为INFO，过滤掉DEBUG日志
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)


def filter_points_in_bbox(points, bbox_min, bbox_max, max_points=10000):
    """
    筛选出在指定边界框内的点云，并随机降采样到指定数量
    
    参数:
        points: 形状为(n, 3)的张量，表示点云
        bbox_min: 形状为(3,)的张量，表示边界框的最小坐标 [x_min, y_min, z_min]
        bbox_max: 形状为(3,)的张量，表示边界框的最大坐标 [x_max, y_max, z_max]
        max_points: 降采样后的最大点数，默认为10000
        
    返回:
        在边界框内且降采样后的点云，以及对应的索引掩码
    """
    # 检查点是否在边界框内（所有维度都要在范围内）
    min_mask = torch.all(points >= bbox_min, dim=1)
    max_mask = torch.all(points <= bbox_max, dim=1)
    
    # 组合掩码
    mask = min_mask & max_mask
    
    # 筛选点
    filtered_points = points[mask]
    
    # 随机降采样到10000个点
    num_points = filtered_points.shape[0]
    if num_points > max_points:
        # 生成随机索引
        perm = torch.randperm(num_points, device=filtered_points.device)
        indices = perm[:max_points]
        
        # 获取降采样后的点
        sampled_points = filtered_points[indices]
        
        # 更新掩码
        # 先找出原始掩码中为True的位置
        true_indices = torch.where(mask)[0]
        
        # 创建一个新的掩码，全部为False
        new_mask = torch.zeros_like(mask)
        
        # 将被选中的索引在新掩码中设为True
        selected_original_indices = true_indices[indices]
        new_mask[selected_original_indices] = True
        
        return sampled_points, new_mask
    else:
        # 如果点数不足max_points，返回所有点
        return filtered_points, mask

def remove_close_points(points_a, points_b, threshold, batch_size=2000):
    """
    从点云a中移除与点云b中任何点的距离小于阈值的点，通过分批处理b点云来避免内存溢出
    
    参数:
        points_a: 形状为(n, 3)的张量，表示密集点云a
        points_b: 形状为(m, 3)的张量，表示稀疏点云b
        threshold: 距离阈值
        batch_size: 每批处理的点云b的点数
        
    返回:
        移除了靠近b点的a点云，以及对应的保留索引掩码
    """
    n = points_a.shape[0]
    m = points_b.shape[0]
    device = points_a.device
    threshold_squared = threshold ** 2
    
    # 初始化保留掩码（开始假设所有点都保留）
    keep_mask = torch.ones(n, dtype=torch.bool, device=device)
    
    # 分批处理点云b
    for i in range(0, m, batch_size):
        # 获取当前批次
        end_idx = min(i + batch_size, m)
        batch_points_b = points_b[i:end_idx]
        
        # 计算点云a到当前批次b点的距离
        a_expanded = points_a.unsqueeze(1)             # shape: (n, 1, 3)
        b_expanded = batch_points_b.unsqueeze(0)       # shape: (1, batch_size, 3)
        
        # 计算距离平方
        dist_squared = torch.sum((a_expanded - b_expanded) ** 2, dim=2)  # shape: (n, batch_size)
        
        # 找出距离小于阈值的a点
        close_to_any_b = torch.any(dist_squared <= threshold_squared, dim=1)  # shape: (n,)
        
        # 更新掩码，排除那些距离太近的点
        keep_mask = keep_mask & (~close_to_any_b)
    
    # 返回过滤后的点云和掩码
    filtered_points_a = points_a[keep_mask]
    
    return filtered_points_a, keep_mask


import torch._dynamo.config
# torch._dynamo.config.force_parameter_static_shapes = False  

def generate_index_list(folder_path, num_samples=40):
    """
    根据文件夹中的文件数量生成索引列表

    参数:
        folder_path (str): 文件夹路径
        num_samples (int): 希望采样的文件数量（默认10）

    返回:
        list: 生成的索引列表
    """
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在")
        return []

    # 获取文件夹中的文件列表（不包含子文件夹）
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    num_files = len(files)

    # 如果文件数量小于等于采样数，返回完整索引
    if num_files <= num_samples:
        return list(range(num_files))

    return np.linspace(0, num_files - 1, num_samples, dtype=int).tolist()



# ##########################################################################
# ######## The following code is for diffusion: StableVirtualCamera ##########
# ##########################################################################

def load_diffusion_models(device="cuda:0"):

    from StableVirtualCamera.seva.model import SGMWrapper
    from StableVirtualCamera.seva.modules.autoencoder import AutoEncoder
    from StableVirtualCamera.seva.modules.conditioner import CLIPConditioner
    from StableVirtualCamera.seva.sampling import DDPMDiscretization, DiscreteDenoiser
    from StableVirtualCamera.seva.utils import load_model
    
    # 检查是否可以使用torch nightly
    # try:
    #     import torch._dynamo
    #     IS_TORCH_NIGHTLY = True
    # except:
    IS_TORCH_NIGHTLY = False
    
    if IS_TORCH_NIGHTLY:
        COMPILE = True
        os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
        os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    else:
        COMPILE = False
    
    # load_model
    MODEL = SGMWrapper(load_model(device=device, verbose=True).eval())
    AE = AutoEncoder(chunk_size=1).to(device)
    CONDITIONER = CLIPConditioner().to(device)
    DISCRETIZATION = DDPMDiscretization()
    DENOISER = DiscreteDenoiser(discretization=DISCRETIZATION, num_idx=1000, device=device)

    print("diffusion models loaded!")
    
    # Imformation version
    VERSION_DICT = {
        "H": 576,
        "W": 576,
        "T": 21,
        "C": 4,
        "f": 8,
        "options": {},
    }
    
    if COMPILE:
        MODEL = torch.compile(MODEL)
        CONDITIONER = torch.compile(CONDITIONER)
        AE = torch.compile(AE)
    
    return {
        "model": MODEL,
        "ae": AE,
        "conditioner": CONDITIONER,
        "discretization": DISCRETIZATION,
        "denoiser": DENOISER,
        "version_dict": VERSION_DICT,
        "compile": COMPILE
    }
def generate_trajectory_from_keyframes(
        keyframe_c2ws: torch.Tensor, 
        keyframe_indices: list[int] | None = None, 
        num_frames: int | None = None,
        scene_center: torch.Tensor | None = None,     # ← 新增，可传点云均值
) -> torch.Tensor:
    """
    根据关键帧位置 + look‑at 重新计算整条轨迹（OpenCV‑c2w）。

    - 平移：线性 / 按边长加权插值
    - 旋转：每帧用 look‑at 重新计算，始终面向 scene_center
    """
    import numpy as np
    import torch
    
    # ---------- 0. 前置 ----------
    device, dtype = keyframe_c2ws.device, keyframe_c2ws.dtype
    kf_num = len(keyframe_c2ws)
    
    # 去掉重复尾帧
    if kf_num >= 2 and torch.allclose(keyframe_c2ws[0], keyframe_c2ws[-1]):
        keyframe_c2ws = keyframe_c2ws[:-1]
        kf_num -= 1
    
    # 确定关键帧索引
    if num_frames is None:
        if keyframe_indices is not None:
            num_frames = max(keyframe_indices) + 1
        else:
            num_frames = kf_num * 10
    if keyframe_indices is None:
        keyframe_indices = (
            np.linspace(0, num_frames, kf_num, endpoint=False).astype(int).tolist()
        )
    
    # scene_center 默认用关键帧平移的几何中心
    if scene_center is None:
        scene_center = keyframe_c2ws[:, :3, 3].mean(0)
    
    # ---------- 1. 预计算平移插值（附带闭环） ----------
    pos_kf = keyframe_c2ws[:, :3, 3]
    
    # 边长
    edges = torch.cat([pos_kf[1:], pos_kf[:1]]) - pos_kf
    seg_len = edges.norm(dim=-1)
    seg_cum = torch.cumsum(seg_len, 0)
    total_len = seg_cum[-1].item()
    
    # 为每条边分配帧数（按长度比例，至少 1 帧）
    base_counts = (seg_len / total_len * num_frames).round().long()
    # 修正四舍五入导致的总和偏差
    diff = num_frames - base_counts.sum()
    if diff != 0:
        base_counts[torch.argmax(seg_len)] += diff
    
    # ---------- 2. 逐段插值 + look‑at ----------
    all_c2ws = torch.zeros((num_frames, 4, 4), dtype=dtype, device=device)
    idx = 0
    for i in range(kf_num):
        j = (i + 1) % kf_num
        cnt = base_counts[i]
        start_p, end_p = pos_kf[i], pos_kf[j]
        
        for t in range(cnt):
            alpha = t / cnt
            pos = (1 - alpha) * start_p + alpha * end_p
            
            # ----- look‑at 计算旋转 (OpenCV: Y向下, Z向前朝向场景) -----
            z_axis = scene_center - pos  # 在OpenCV中，Z轴朝向场景中心
            z_axis = z_axis / (z_axis.norm() + 1e-8)   
            
            # OpenCV中Y轴向下，定义世界坐标系的up向量
            world_up = torch.tensor([0., -1., 0.], device=device)
            
            # 修正：计算右向量，应该是 up × forward = right
            x_axis = torch.cross(world_up, z_axis)  # 正确的叉乘顺序
            
            if x_axis.norm() < 1e-6:  # 处理当相机方向与up向量平行时的特殊情况
                world_up = torch.tensor([0., 0., 1.], device=device)
                x_axis = torch.cross(world_up, z_axis)  # 正确的叉乘顺序
            
            x_axis = x_axis / x_axis.norm()
            
            # 修正：计算下向量，应该是 forward × right = down
            y_axis = torch.cross(z_axis, x_axis)  # 正确的叉乘顺序
            y_axis = y_axis / (y_axis.norm() + 1e-8)
            
            # 在OpenCV中，相机坐标系为[right, down, forward]
            R_cv = torch.stack([x_axis, y_axis, z_axis], 1)
            
            c2w = torch.eye(4, dtype=dtype, device=device)
            c2w[:3, :3] = R_cv
            c2w[:3,  3] = pos
            all_c2ws[idx] = c2w
            idx += 1
    
    # 若因为四舍五入少 1 帧，把最后一帧补上
    if idx < num_frames:
        all_c2ws[idx:] = all_c2ws[idx-1]
    
    return all_c2ws


# def calculate_box_trajectories_opengl(x_min, x_max, full_pc):
#     """
#     生成 5 个相机位姿的 OpenGL **view matrix (world -> camera)**。
    
#     - 轨迹：沿 X-Z 平面围绕点云中心走一个矩形环 (首尾闭合)。
#     - 坐标系：右手系，+Y 向上，摄像机前向 = -Z。
    
#     Args
#     ----
#     x_min, x_max : (3,) Tensor/list
#         包围盒最小 / 最大角点。
#     full_pc      : (N,3) Tensor
#         整个场景点云（需与 x_min/x_max 同一坐标系）。
    
#     Returns
#     -------
#     trajectory_w2c : (5, 4, 4) Tensor
#         每一帧的 **world-to-camera** 4×4 齐次矩阵，可直接喂给 OpenGL。
#     """
#     x_min, x_max = torch.as_tensor(x_min, device=full_pc.device), \
#                    torch.as_tensor(x_max, device=full_pc.device)

#     # 1. 过滤点云到包围盒并取中心
#     mask = ((full_pc >= x_min) & (full_pc <= x_max)).all(dim=-1)
#     center = full_pc[mask].mean(0)
#     center_np = center.cpu().numpy()

#     # 2. 盒子 4 个顶点（Y 作 ±1 的轻微偏置以避免退化）
#     cy = center[1]
#     vertices = torch.stack([
#     torch.tensor([x_min[0], cy + 4, x_min[2]],
#                  device=full_pc.device, dtype=full_pc.dtype),
#     torch.tensor([x_max[0], cy - 4, x_min[2]],
#                  device=full_pc.device, dtype=full_pc.dtype),
#     torch.tensor([x_max[0], cy + 4, x_max[2]],
#                  device=full_pc.device, dtype=full_pc.dtype),
#     torch.tensor([x_min[0], cy - 4, x_max[2]],
#                  device=full_pc.device, dtype=full_pc.dtype),
#     ])
#     order = [0, 1, 2, 3, 0]          # 前左→前右→后右→后左→回到前左
#     trajectory = torch.zeros((5, 4, 4), device=full_pc.device)

#     for i, idx in enumerate(order):
#         pos = vertices[idx]
#         pos_np = pos.cpu().numpy()

#         # ---- 求各轴 -----------------------------------------------------------------
#         z_axis = center_np - pos_np                # 相机 -> 目标 (PyTorch3D 为 +Z)
#         z_axis /= np.linalg.norm(z_axis) + 1e-8
#         z_axis = -z_axis                           # OpenGL 需要 -Z 指向前方

#         world_up = np.array([0.0, 1.0, 0.0])
#         x_axis = np.cross(world_up, z_axis)
#         if np.linalg.norm(x_axis) < 1e-6:          # 退化：视线接近世界上方向
#             world_up = np.array([1.0, 0.0, 0.0])
#             x_axis = np.cross(world_up, z_axis)
#         x_axis /= np.linalg.norm(x_axis)
#         y_axis = np.cross(z_axis, x_axis)

#         R_gl = torch.tensor([x_axis, y_axis, z_axis],
#                     device=full_pc.device,
#                     dtype=full_pc.dtype).T

#         # ---- 组装 world->camera view matrix ----------------------------------------
#         w2c = torch.zeros((4, 4), device=full_pc.device)
#         w2c[:3, :3] = R_gl.T
#         w2c[:3, 3] = -(R_gl.T @ pos)
#         w2c[3, 3] = 1.0

#         trajectory[i] = w2c

#     return trajectory


import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union
import roma  # 需要安装: pip install roma


def orbit_from_known_poses(
    point_cloud: Union[torch.Tensor, np.ndarray],  # 点云 (N, 3)
    known_poses: Union[torch.Tensor, np.ndarray],  # 已知相机poses (M, 4, 4)
    num_frames: int = 80,
    device: str = "cpu"
) -> torch.Tensor:
    """
    根据点云和已知相机poses生成orbit轨迹
    
    策略：
    1. 计算点云中心
    2. 找到已知poses中距离点云中心最远的距离作为轨道半径
    3. 选择最靠近点云中心y平面的pose作为起始点
    4. 在该y平面上绕点云中心旋转
    
    Args:
        point_cloud: 点云坐标 (N, 3)
        known_poses: 已知相机poses (M, 4, 4) - camera-to-world矩阵
        num_frames: 轨迹帧数
        device: 计算设备
        
    Returns:
        trajectory: (num_frames, 4, 4) 的相机轨迹
    """
    
    # 1. 计算点云中心
    cloud_center = point_cloud.mean(dim=0)  # [3]
    
    # 2. 提取已知poses的相机位置 (假设是c2w矩阵)
    camera_positions = known_poses[:, :3, 3]  # (M, 3)
    
    # 计算所有相机到点云中心的距离
    distances = torch.norm(camera_positions - cloud_center, dim=1)  # (M,)
    max_distance = distances.max().item()
    
    # 3. 找到最靠近点云中心y平面的pose
    y_center = cloud_center[1].item()
    y_distances = torch.abs(camera_positions[:, 1] - y_center)  # y方向距离
    closest_idx = y_distances.argmin()
    closest_pose = camera_positions[closest_idx]  # [3]
    
    # 4. 构造起始位置：保持该pose的y坐标，但调整到最远距离
    start_y = closest_pose[1].item()
    
    # 在xz平面上，从点云中心出发，距离为max_distance的位置作为起始点
    start_position = [
        cloud_center[0].item() + max_distance+5,  # x方向偏移
        cloud_center[1],                                # 保持最接近的y坐标
        cloud_center[2].item()                  # z方向对齐中心
    ]
    
    # 5. 生成orbit轨迹
    return generate_orbit_trajectory(
        start_camera_position=start_position,
        look_at_point=cloud_center.tolist(),
        num_frames=num_frames,
        up_direction=[0, -1, 0],  # 保持OpenCV坐标系
        device=device
    )


def generate_orbit_trajectory(
    # 必需参数
    start_camera_position: Union[torch.Tensor, np.ndarray, list],  # 起始相机位置 [x, y, z]
    look_at_point: Union[torch.Tensor, np.ndarray, list],          # 注视点 [x, y, z]
    num_frames: int,                                               # 轨迹帧数
    
    # 可选参数 - 与原代码保持一致
    up_direction: Optional[Union[torch.Tensor, np.ndarray, list]] = None,  # 上方向 [x, y, z]
    orbit_degree: float = 360.0,                                   # 轨道角度范围(度)
    clockwise: bool = True,                                        # 是否顺时针
    radius_scale: float = 1.0,                                     # 半径缩放因子
    height_offset: float = 0.0,                                    # 高度偏移
    include_endpoint: bool = False,                                # 是否包含终点
    device: str = "cpu",                                           # 计算设备
) -> torch.Tensor:
    """
    生成orbit(环绕)相机轨迹，输出格式与calculate_box_trajectories_opencv一致
    
    Args:
        start_camera_position: 起始相机位置，形状为 [3] 的数组
        look_at_point: 相机注视的目标点，形状为 [3] 的数组  
        num_frames: 生成的轨迹帧数
        up_direction: 上方向向量，默认为 [0, -1, 0] (与OpenCV坐标系一致)
        orbit_degree: 轨道角度范围，360度为完整圆圈
        clockwise: 是否顺时针旋转
        radius_scale: 轨道半径缩放因子
        height_offset: 垂直方向偏移
        include_endpoint: 是否包含轨迹终点(避免首尾重复)
        device: PyTorch计算设备
        
    Returns:
        trajectory: 相机轨迹矩阵，形状为 (num_frames, 4, 4)
                   每个4x4矩阵是camera-to-world变换矩阵
        
    Example:
        >>> # 基本用法 - 生成完整圆形轨迹
        >>> trajectory = generate_orbit_trajectory(
        ...     start_camera_position=[3, 0, 0],
        ...     look_at_point=[0, 0, 0], 
        ...     num_frames=80
        ... )
        >>> print(f"轨迹形状: {trajectory.shape}")  # (80, 4, 4)
        
        >>> # 高级用法 - 3/4圆弧轨迹
        >>> trajectory = generate_orbit_trajectory(
        ...     start_camera_position=[2, 1, 1],
        ...     look_at_point=[0, 0, 0],
        ...     num_frames=60,
        ...     up_direction=[0, -1, 0],  # Y轴向下(OpenCV约定)
        ...     orbit_degree=270,         # 3/4圆
        ...     clockwise=False,          # 逆时针
        ...     radius_scale=1.5,         # 1.5倍半径
        ...     height_offset=0.5         # 向上偏移0.5单位
        ... )
    """
    
    # 输入验证和类型转换
    def to_tensor(x, name):
        if isinstance(x, (list, np.ndarray)):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        elif isinstance(x, torch.Tensor):
            x = x.float().to(device)
        else:
            raise TypeError(f"{name} must be tensor, numpy array or list")
        return x
    
    start_pos = to_tensor(start_camera_position, "start_camera_position")
    look_at = to_tensor(look_at_point, "look_at_point")
    
    if start_pos.shape != torch.Size([3]) or look_at.shape != torch.Size([3]):
        raise ValueError("start_camera_position and look_at_point must have shape [3]")
    
    # 默认使用OpenCV坐标系约定：Y轴向下
    if up_direction is None:
        up_direction = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=torch.float32)
    else:
        up_direction = to_tensor(up_direction, "up_direction")
        if up_direction.shape != torch.Size([3]):
            raise ValueError("up_direction must have shape [3]")
    
    # 标准化up方向
    up_direction = F.normalize(up_direction, dim=0)
    
    # 计算参考位置(应用偏移和缩放)
    ref_position = start_pos + up_direction * height_offset
    radius_vector = ref_position - look_at
    ref_position = look_at + radius_vector * radius_scale
    
    # 生成角度序列
    if include_endpoint:
        thetas = torch.linspace(0.0, torch.pi * orbit_degree / 180, num_frames, device=device)
    else:
        thetas = torch.linspace(0.0, torch.pi * orbit_degree / 180, num_frames + 1, device=device)[:-1]
    
    if not clockwise:
        thetas = -thetas
    
    # 生成轨道位置 - 使用罗德里格斯旋转公式
    positions = (
        torch.einsum(
            "nij,j->ni",
            roma.rotvec_to_rotmat(thetas[:, None] * up_direction[None]),  # 旋转矩阵序列
            ref_position - look_at,  # 从注视点到参考位置的向量
        )
        + look_at  # 平移回世界坐标系
    )
    
    # 构建camera-to-world变换矩阵
    trajectory = torch.zeros((num_frames, 4, 4), device=device, dtype=torch.float32)
    
    for i in range(num_frames):
        pos = positions[i]
        
        # 计算相机朝向 (从相机指向注视点)
        look_dir = look_at - pos
        look_dir = look_dir / look_dir.norm()
        
        # 构建相机坐标系 (与calculate_box_trajectories_opencv保持一致)
        z_axis = look_dir  # 相机前向轴
        world_up = up_direction
        
        # 计算右向轴 (x轴)
        x_axis = torch.cross(world_up, z_axis)
        if x_axis.norm() < 1e-6:
            # 处理up方向与look方向平行的情况
            world_up = torch.tensor([0., 0., 1.], device=device, dtype=torch.float32)
            x_axis = torch.cross(world_up, z_axis)
        x_axis = x_axis / x_axis.norm()
        
        # 计算上向轴 (y轴) - 确保正交性
        y_axis = torch.cross(z_axis, x_axis)
        y_axis = y_axis / y_axis.norm()
        
        # 构建旋转矩阵 [x_axis, y_axis, z_axis]
        R = torch.stack([x_axis, y_axis, z_axis], dim=1)
        
        # 构建4x4变换矩阵 (camera-to-world)
        c2w = torch.eye(4, device=device, dtype=torch.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = pos
        
        trajectory[i] = c2w
    
    return trajectory

def orbit_from_bounding_box(
    x_min: Union[torch.Tensor, np.ndarray, list],
    x_max: Union[torch.Tensor, np.ndarray, list], 
    num_frames: int = 80,
    height_ratio: float = 0.0,  # 相对于box高度的比例
    radius_ratio: float = 1.5,  # 相对于box对角线的比例
    device: str = "cpu"
) -> torch.Tensor:
    """
    根据场景包围盒生成orbit轨迹
    
    Args:
        x_min: 包围盒最小坐标 [x, y, z]
        x_max: 包围盒最大坐标 [x, y, z]
        num_frames: 轨迹帧数
        height_ratio: 相机高度相对于box高度的比例 (0.0=box中心高度)
        radius_ratio: 轨道半径相对于box对角线长度的比例
        device: 计算设备
        
    Returns:
        trajectory: (num_frames, 4, 4) 的相机轨迹
    """
    def to_tensor(x):
        if isinstance(x, (list, np.ndarray)):
            return torch.tensor(x, dtype=torch.float32, device=device)
        elif isinstance(x, torch.Tensor):
            return x.float().to(device)
        else:
            raise TypeError("Input must be tensor, numpy array or list")
    
    x_min = to_tensor(x_min)
    x_max = to_tensor(x_max)
    
    # 计算包围盒中心和尺寸
    box_center = (x_min + x_max) / 2
    box_size = x_max - x_min
    box_diagonal = torch.norm(box_size)
    
    # 计算相机起始位置
    camera_height = box_center[1] + box_size[1] * height_ratio
    camera_radius = box_diagonal * radius_ratio / 2
    
    start_position = [
        box_center[0].item() + camera_radius,
        camera_height.item(),
        box_center[2].item()
    ]
    
    look_at = box_center.tolist()
    
    return generate_orbit_trajectory(
        start_camera_position=start_position,
        look_at_point=look_at,
        num_frames=num_frames,
        device=device
    )





def rotation_matrix_to_quaternion(R):
    """
    将旋转矩阵转换为四元数 [w, x, y, z]
    与GUI代码中的格式保持一致
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    # 转换为splines库所需的UnitQuaternion格式
    return splines.quaternion.UnitQuaternion.from_unit_xyzw(np.array([qx, qy, qz, qw]))


def calculate_box_trajectories_opencv(
    x_min, x_max, full_pc,
    num_frames=80,
    tension=0.0,
    loop=True,
    add_vertical_oscillation=True
):
    """
    生成既平滑又匀速的相机轨迹
    结合样条插值的平滑性和等距离采样的匀速性
    
    Args:
        x_min, x_max: 包围盒的最小和最大坐标
        full_pc: 完整的点云数据
        num_frames: 插值生成的总帧数
        tension: 样条张力参数 (0.0 = 平滑, 1.0 = 紧绷)
        loop: 是否闭合轨迹
        add_vertical_oscillation: 是否添加垂直波动
    
    Returns:
        interpolated_trajectory: [num_frames, 4, 4] 的相机轨迹
    """
    x_min, x_max = (torch.as_tensor(a, device=full_pc.device, dtype=full_pc.dtype) for a in (x_min, x_max))
    
    # 1. 计算场景中心
    box_center = (x_min + x_max) / 2
    mask_all = ((full_pc >= x_min) & (full_pc <= x_max)).all(dim=-1)
    if mask_all.sum().item() > 0:
        pc_center = full_pc[mask_all].mean(0)
        scene_center = box_center
        scene_center[1] = pc_center[1]
    else:
        scene_center = box_center
        
    # 2. 定义矩形相机路径顶点
    cy = box_center[1]
    vertices = torch.stack([
        torch.tensor([x_min[0], cy, x_min[2]], device=full_pc.device, dtype=full_pc.dtype),
        torch.tensor([x_max[0], cy, x_min[2]], device=full_pc.device, dtype=full_pc.dtype),
        torch.tensor([x_max[0], cy, x_max[2]], device=full_pc.device, dtype=full_pc.dtype),
        torch.tensor([x_min[0], cy, x_max[2]], device=full_pc.device, dtype=full_pc.dtype),
    ])
    
    # 3. 构造关键帧
    if loop:
        order = [0, 1, 2, 3, 0]
    else:
        order = [0, 1, 2, 3]
        
    keyframe_positions = []
    keyframe_quaternions = []
    
    for i, vidx in enumerate(order):
        pos = vertices[vidx]
        
        # 计算相机朝向
        look_dir = scene_center - pos
        look_dir = look_dir / look_dir.norm()
        
        # 构建相机坐标系
        z_axis = look_dir
        world_up = torch.tensor([0., -1., 0.], device=full_pc.device, dtype=full_pc.dtype)
        x_axis = torch.cross(world_up, z_axis)
        
        if x_axis.norm() < 1e-6:
            world_up = torch.tensor([0., 0., 1.], device=full_pc.device, dtype=full_pc.dtype)
            x_axis = torch.cross(world_up, z_axis)
            
        x_axis = x_axis / x_axis.norm()
        y_axis = torch.cross(z_axis, x_axis)
        y_axis = y_axis / y_axis.norm()
        
        R = torch.stack([x_axis, y_axis, z_axis], dim=1)
        quat = rotation_matrix_to_quaternion(R.cpu().numpy())
        
        keyframe_positions.append(pos.cpu().numpy())
        keyframe_quaternions.append(quat)
    
    print(f"生成了 {len(keyframe_positions)} 个关键帧")
    
    # 4. 创建高密度样条轨迹用于距离计算
    # 先用样条生成一个高密度的平滑轨迹
    high_density_frames = 1000  # 用于距离计算的高密度采样
    
    position_spline = splines.KochanekBartels(
        keyframe_positions,
        tcb=(tension, 0.0, 0.0),
        endconditions="closed" if loop else "natural",
    )
    
    orientation_spline = splines.quaternion.KochanekBartels(
        keyframe_quaternions,
        tcb=(tension, 0.0, 0.0),
        endconditions="closed" if loop else "natural",
    )
    
    # 5. 生成高密度轨迹点用于距离计算
    num_keyframes = len(keyframe_positions)
    max_spline_t = num_keyframes - 1 if not loop else num_keyframes
    
    high_density_t = np.linspace(0, max_spline_t, high_density_frames)
    high_density_positions = []
    
    for t in high_density_t:
        pos = position_spline.evaluate(t)
        high_density_positions.append(pos)
    
    high_density_positions = np.array(high_density_positions)
    
    # 6. 计算累积弧长
    cumulative_distances = [0.0]
    for i in range(1, len(high_density_positions)):
        dist = np.linalg.norm(high_density_positions[i] - high_density_positions[i-1])
        cumulative_distances.append(cumulative_distances[-1] + dist)
    
    total_length = cumulative_distances[-1]
    cumulative_distances = np.array(cumulative_distances)
    
    print(f"轨迹总长度: {total_length}")
    
    # 7. 创建弧长到样条参数t的映射
    # 使用插值器从弧长映射到样条参数t
    arc_length_to_t_interpolator = PchipInterpolator(
        cumulative_distances, 
        high_density_t
    )
    
    # 8. 等弧长采样生成最终轨迹
    interpolated_trajectory = torch.zeros((num_frames, 4, 4), device=full_pc.device, dtype=full_pc.dtype)
    
    for i in range(num_frames):
        # 计算当前帧对应的弧长位置
        target_arc_length = (i / (num_frames - 1)) * total_length
        
        # 从弧长映射到样条参数t
        spline_t = float(arc_length_to_t_interpolator(target_arc_length))
        
        # 从样条获取位置和旋转
        pos_interp = position_spline.evaluate(spline_t)
        quat_interp = orientation_spline.evaluate(spline_t)
        
        pos_interp = torch.tensor(pos_interp, device=full_pc.device, dtype=full_pc.dtype)
        
        # 添加垂直波动（可选）
        if add_vertical_oscillation:
            vertical_oscillation = torch.sin(torch.tensor(i * 2 * 3.14159 / num_frames)) * 0.1
            pos_interp[1] += vertical_oscillation
        
        # 重新计算朝向（确保始终看向场景中心）
        look_dir = scene_center - pos_interp
        look_dir = look_dir / look_dir.norm()
        
        z_axis = look_dir
        world_up = torch.tensor([0., -1., 0.], device=full_pc.device, dtype=full_pc.dtype)
        
        x_axis = torch.cross(world_up, z_axis)
        if x_axis.norm() < 1e-6:
            world_up = torch.tensor([0., 0., 1.], device=full_pc.device, dtype=full_pc.dtype)
            x_axis = torch.cross(world_up, z_axis)
            
        x_axis = x_axis / x_axis.norm()
        y_axis = torch.cross(z_axis, x_axis)
        y_axis = y_axis / y_axis.norm()
        
        R = torch.stack([x_axis, y_axis, z_axis], dim=1)
        
        # 构建变换矩阵
        c2w_interp = torch.eye(4, device=full_pc.device, dtype=full_pc.dtype)
        c2w_interp[:3, :3] = R
        c2w_interp[:3, 3] = pos_interp
        
        interpolated_trajectory[i] = c2w_interp
    
    return interpolated_trajectory

import os
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn.functional as F

def load_camera_data_from_pt(
    folder_path: str,
    indices: List[int],
    K: torch.Tensor,  # 原始内参矩阵
    target_size: Tuple[int, int] = (1152, 576),
    convert_from_pytorch3d: bool = True,
    T_is_camera_center: bool = False,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    pixel_coords: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    将存储于 .pt 文件中的 PyTorch3D 相机参数转换为 OpenCV 坐标系下的 c2w，
    并返回处理后的图像、内参、像素坐标等信息。
    """
    input_imgs, input_c2ws, input_Ks = [], [], []
    new_pixel_coords = None

    if pixel_coords is not None:
        pixel_coords = pixel_coords.to(device)

    # PyTorch3D ➜ OpenCV 坐标变换矩阵（翻转Y和Z轴）
    S = torch.diag(torch.tensor([1., -1., -1., 1.], device=device))

    for i, idx in enumerate(sorted(indices)):
        pt_path = os.path.join(folder_path, f"{idx}.pt")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"找不到文件: {pt_path}")

        data = torch.load(pt_path, map_location=device)
        R = torch.tensor(data['R'], dtype=torch.float32, device=device)  # [3, 3]
        T = torch.tensor(data['T'], dtype=torch.float32, device=device)  # [3]

        # 如果 T 是相机中心 C，则转换为 t_w2c = -R @ C；否则直接使用 T 作为平移
        if T_is_camera_center:
            C = T.view(3, 1)
            t_w2c = (-R @ C).view(3)
        else:
            t_w2c = T

        # 构建 PyTorch3D 坐标系下的 w2c 变换矩阵
        w2c = torch.eye(4, dtype=torch.float32, device=device)
        w2c[:3, :3] = R
        w2c[:3, 3] = t_w2c

        # 图像读取和缩放
        img = data['rgb'].squeeze()
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.permute(1, 2, 0)  # [H, W, 3]

        h0, w0 = img.shape[:2]
        scale_w, scale_h = target_size[0] / w0, target_size[1] / h0

        # 缩放像素坐标（注意方向：x=列=width，y=行=height）
        if pixel_coords is not None and i == 0:
            new_pixel_coords = pixel_coords.clone().float()
            new_pixel_coords[:, 0] *= scale_w  # x direction (width)
            new_pixel_coords[:, 1] *= scale_h  # y direction (height)

        # 图像插值缩放
        img = F.interpolate(
            img.permute(2, 0, 1).unsqueeze(0),
            size=(target_size[1], target_size[0]),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)

        if img.max() > 1:
            img = img / 255.0
        input_imgs.append(img)

        # 坐标系变换：P3D ➜ OpenCV，只对 c2w 变换
        if convert_from_pytorch3d:
            w2c_cv = S @ w2c           # ✅ S 乘在左侧
            c2w_cv = torch.inverse(w2c_cv)
        else:
            c2w_cv = torch.inverse(w2c)

        input_c2ws.append(c2w_cv)

        # 缩放内参矩阵
        K_scaled = K.clone().to(device)
        K_scaled[0, 0] *= scale_w  # fx
        K_scaled[1, 1] *= scale_h  # fy
        K_scaled[0, 2] *= scale_w  # cx
        K_scaled[1, 2] *= scale_h  # cy

        # ✅ 归一化到 [0, 1]：除以 W, H
        K_scaled[0, :] /= target_size[0]
        K_scaled[1, :] /= target_size[1]
        input_Ks.append(K_scaled)

    result = {
        "input_imgs": torch.stack(input_imgs),
        "input_c2ws": torch.stack(input_c2ws),
        "input_Ks":   torch.stack(input_Ks)
    }
    if new_pixel_coords is not None:
        result["new_pixel_coords"] = new_pixel_coords

    return result



import copy
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
import os
import os.path as osp
from StableVirtualCamera.seva.eval import infer_prior_stats, run_one_scene, chunk_input_and_test

    # if points is not None:
    #     all_points = points.cpu().numpy()
    #     point_chunks = [points.shape[0]]
    #     point_indices = []  # 没有分割点

    #     input_c2ws_np = input_c2ws.cpu().numpy()
    #     target_c2ws_np = keyframe_c2ws.cpu().numpy()

    #     normalized_c2ws, normalized_points, transform_matrix = normalize_scene(
    #         input_c2ws_np,
    #         all_points,
    #         camera_center_method="poses",
    #     )

    #     normalized_points = np.split(normalized_points, point_indices, 0)

    #     scene_scale = np.median(
    #         np.ptp(np.concatenate([normalized_c2ws[:, :3, 3], *normalized_points], 0), -1)
    #     )

    #     normalized_c2ws[:, :3, 3] /= scene_scale
    #     normalized_points = [point / scene_scale for point in normalized_points]
        
    #     # 使用相同的变换矩阵处理目标相机位姿
    #     # 根据normalize_scene的实现，应使用transform_cameras函数应用相同的变换
    #     normalized_target_c2ws = transform_cameras(transform_matrix, target_c2ws_np)
        
    #     # 同样对目标相机位置应用相同的缩放
    #     normalized_target_c2ws[:, :3, 3] /= scene_scale

def generate_images_img2video(
    diffusion_models,
    input_c2ws,
    input_imgs,
    input_Ks,
    keyframe_c2ws,
    points=None,
    save_dir=None,
    device=None,
):
    """
    使用Stable Virtual Camera(SEVA)生成新视角图像
    
    参数:
        diffusion_models: 包含模型的字典，需包含'model', 'ae', 'conditioner', 'denoiser'
        input_c2ws: 输入相机的c2w矩阵，形状为[num_inputs, 4, 4]
        input_imgs: 输入图像，形状为[num_inputs, H, W, 3]或[num_inputs, 3, H, W]
        input_Ks: 输入相机的内参矩阵，形状为[num_inputs, 3, 3]
        keyframe_c2ws: 关键帧相机轨迹，形状为[num_keyframes, 4, 4]
        points: 点云，tensor形状为[n, 3]
        point_colors: 点云颜色，与points形状匹配
        scene_scale: 场景缩放系数，None则自动计算
        seed: 随机种子
        chunk_strategy: 分块策略，"interp-gt"或"interp"
        cfg: CFG值，控制生成图像的保真度
        camera_scale: 相机缩放系数
        options: 额外渲染选项
        abort_event: 用于中止渲染的事件
        save_dir: 保存目录
        num_steps: 渲染步数
        convert_coordinate_system: 是否将相机从OpenCV转为OpenGL坐标系
        
    返回:
        first_pass_video: 第一阶段生成的视频路径
        second_pass_video: 第二阶段生成的视频路径
        save_dir: 保存目录
    """
    import copy
    import os
    import threading
    from datetime import datetime
    import numpy as np
    import torch
    import torch.nn.functional as F
    from StableVirtualCamera.seva.eval import infer_prior_stats, chunk_input_and_test, run_one_scene
    from StableVirtualCamera.seva.geometry import get_default_intrinsics, normalize_scene, DEFAULT_FOV_RAD, transform_cameras

    # 14. 设置保存目录
    if save_dir is None:
        save_dir = f"work_dirs/generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to {save_dir}")

    # input_imgs: [N, H, W, 3] or [N, 3, H, W]
    if input_imgs.ndim == 4 and input_imgs.shape[-1] == 3:
        H, W = input_imgs.shape[1:3]
    elif input_imgs.ndim == 4 and input_imgs.shape[1] == 3:
        H, W = input_imgs.shape[2:4]
    else:
        raise ValueError("input_imgs shape must be [N,H,W,3] or [N,3,H,W]")
    
    num_inputs = len(input_imgs)
    num_targets = len(keyframe_c2ws)

    H, W = input_imgs.shape[1:3]

    VERSION_DICT = {
    "H": H,
    "W": W,
    "T": 21,
    "C": 4,
    "f": 8,
    "options": {},
    }

    options = VERSION_DICT["options"]
    options["chunk_strategy"] = "interp-gt" if num_inputs <= 10 else "interp"
    options["video_save_fps"] = 30.0
    options["beta_linear_start"] = 5e-6
    options["log_snr_shift"] = 2.4
    options["guider_types"] = [1, 2]
    options["cfg"] = [float(3.0), 3.0 if num_inputs >= 9 else 2.0]
    options["camera_scale"] = 2.0
    options["num_steps"] = 50
    options["cfg_min"] = 1.2
    options["encoding_t"] = 1
    options["decoding_t"] = 1
    options["num_inputs"] = None
    options["seed"] = 23
    
    seed = options["seed"]

    options["num_prior_frames"] = 15        # 强制20锚点
    options["num_input_semi_dense"] = 1 


    # task == "img2trajvid"

        
    #     # 将规范化后的数据转回tensor
    #     input_c2ws = torch.tensor(normalized_c2ws, dtype=torch.float32, device=device)
    #     keyframe_c2ws = torch.tensor(normalized_target_c2ws, dtype=torch.float32, device=device)
    
    target_Ks = input_Ks[0:1].repeat(num_targets, 1, 1)

    all_c2ws = torch.cat([input_c2ws, keyframe_c2ws], 0)
    all_Ks = torch.cat([input_Ks, target_Ks], 0) * torch.tensor([W, H, 1], device=device)[:, None]

    input_indices = list(range(num_inputs))
    target_indices = list(range(num_inputs, num_inputs + num_targets))

    num_anchors = infer_prior_stats(
        VERSION_DICT["T"],
        num_inputs,
        num_total_frames=num_targets,
        version_dict=VERSION_DICT,
    )

    T = VERSION_DICT["T"]

    anchor_indices = np.linspace(
        num_inputs,
        num_inputs + num_targets - 1,
        num_anchors,
    ).tolist()
    anchor_c2ws = all_c2ws[[round(ind) for ind in anchor_indices]]
    anchor_Ks = all_Ks[[round(ind) for ind in anchor_indices]]

    all_imgs_np = (
        F.pad(input_imgs, (0, 0, 0, 0, 0, 0, 0, num_targets), value=0.0).cpu().numpy()
        * 255.0
    ).astype(np.uint8)
    image_cond = {
        "img": all_imgs_np,
        "input_indices": input_indices,
        "prior_indices": anchor_indices,
    }
    
    # 创建相机条件
    camera_cond = {
        "c2w": all_c2ws,
        "K": all_Ks,
        "input_indices": list(range(num_inputs + num_targets)),
    }

    video_paths1, video_paths2= run_one_scene(
        task="img2trajvid",
        version_dict={
            "H": H,
            "W": W,
            "T": T,
            "C": VERSION_DICT["C"],
            "f": VERSION_DICT["f"],
            "options": options,
        },
        model=diffusion_models["model"],
        ae=diffusion_models["ae"],
        conditioner=diffusion_models["conditioner"],
        denoiser=diffusion_models["denoiser"],
        image_cond=image_cond,
        camera_cond=camera_cond,
        save_path=save_dir,
        use_traj_prior=True,
        traj_prior_c2ws=anchor_c2ws,
        traj_prior_Ks=anchor_Ks,
        seed=seed,
    )

    return video_paths1, video_paths2, save_dir


def generate_images_img2img(
    diffusion_models,
    input_c2ws,
    input_imgs,
    input_Ks,
    keyframe_c2ws,
    points=None,
    save_dir=None,
    device=None,
):
    """
    使用Stable Virtual Camera(SEVA)的img2img任务生成新视角图像
    
    参数:
        diffusion_models: 包含模型的字典，需包含'model', 'ae', 'conditioner', 'denoiser'
        input_c2ws: 输入相机的c2w矩阵，形状为[num_inputs, 4, 4]
        input_imgs: 输入图像，形状为[num_inputs, H, W, 3]或[num_inputs, 3, H, W]
        input_Ks: 输入相机的内参矩阵，形状为[num_inputs, 3, 3]
        keyframe_c2ws: 需要生成的目标相机poses，形状为[num_targets, 4, 4]
        points: 点云，tensor形状为[n, 3] (暂未使用)
        save_dir: 保存目录
        device: 设备
        
    返回:
        video_path: 生成的视频路径 (img2img是单阶段，只返回一个路径)
        save_dir: 保存目录
    """
    import copy
    import os
    from datetime import datetime
    import numpy as np
    import torch
    import torch.nn.functional as F
    from StableVirtualCamera.seva.eval import run_one_scene

    # 设置保存目录
    if save_dir is None:
        save_dir = f"work_dirs/generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to {save_dir}")

    # 解析图像尺寸
    print(f"Input images shape: {input_imgs.shape}")
    print(f"Input images dtype: {input_imgs.dtype}")
    print(f"Input images min/max: {input_imgs.min():.3f}/{input_imgs.max():.3f}")
    
    if input_imgs.ndim == 4 and input_imgs.shape[-1] == 3:
        H, W = input_imgs.shape[1:3]
        # 转换为 [N, 3, H, W] 格式
        input_imgs = input_imgs.permute(0, 3, 1, 2)
    elif input_imgs.ndim == 4 and input_imgs.shape[1] == 3:
        H, W = input_imgs.shape[2:4]
    else:
        raise ValueError(f"input_imgs shape must be [N,H,W,3] or [N,3,H,W], got {input_imgs.shape}")
    
    print(f"After processing - Input images shape: {input_imgs.shape}")
    print(f"Resolved H, W: {H}, {W}")
    
    num_inputs = len(input_imgs)
    num_targets = len(keyframe_c2ws)
    
    print(f"num_inputs: {num_inputs}, num_targets: {num_targets}")

    # 版本配置
    VERSION_DICT = {
        "H": H,
        "W": W,
        "T": 25,
        "C": 4,
        "f": 8,
        "options": {},
    }

    options = VERSION_DICT["options"]
    options["chunk_strategy"] = "gt-nearest"  # img2img使用gt策略
    options["video_save_fps"] = 30.0
    options["beta_linear_start"] = 5e-6
    options["log_snr_shift"] = 2.4
    options["guider_types"] = 1  # img2img使用单一guider
    options["cfg"] = 2.0  # img2img使用单一cfg值
    options["camera_scale"] = 2.0
    options["num_steps"] = 50
    options["cfg_min"] = 1.2
    options["encoding_t"] = 1
    options["decoding_t"] = 1
    options["num_inputs"] = None
    options["seed"] = 23
    
    seed = options["seed"]

    # 准备图像数据 - 需要转换为正确的格式给run_one_scene
    print(f"Preparing image data...")
    
    # 重要发现：run_one_scene期望的numpy数组格式可能不同
    # 让我们检查原始img2trajvid是如何传递数据的
    
    # 方法1：尝试按照run_one_scene内部逻辑，使用列表格式而非numpy数组
    all_imgs_data = []
    
    # 处理输入图像 - 转换为HWC格式的numpy数组
    for i in range(num_inputs):
        img = input_imgs[i]  # [3, H, W]
        img_hwc = img.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        if img_hwc.max() <= 1.0:
            img_hwc = (img_hwc * 255).astype(np.uint8)
        all_imgs_data.append(img_hwc)
        print(f"Input image {i} shape: {img_hwc.shape}")
    
    # 添加目标位置的None占位符
    for i in range(num_targets):
        all_imgs_data.append(None)
    
    print(f"Total images in list: {len(all_imgs_data)}")
    print(f"Input images: {num_inputs}, Target placeholders: {num_targets}")

    # 准备相机参数
    print(f"Preparing camera parameters...")
    print(f"input_Ks shape: {input_Ks.shape}")
    print(f"keyframe_c2ws shape: {keyframe_c2ws.shape}")
    
    target_Ks = input_Ks[0:1].repeat(num_targets, 1, 1)  # 复用第一个相机的内参
    
    all_c2ws = torch.cat([input_c2ws, keyframe_c2ws], 0)
    all_Ks = torch.cat([input_Ks, target_Ks], 0)
    
    # 检查输入内参的数值范围，判断是否需要缩放
    print(f"Sample input K matrix:\n{input_Ks[0]}")
    print(f"fx={input_Ks[0,0,0]:.3f}, fy={input_Ks[0,1,1]:.3f}")
    
    # 如果fx, fy < 10，很可能是归一化的内参，需要恢复到像素坐标
    # 按照原始img2trajvid的做法，确保一致性
    if device is not None:
        all_Ks = all_Ks * torch.tensor([W, H, 1], device=device)[:, None]
    else:
        all_Ks = all_Ks * torch.tensor([W, H, 1])[:, None]
    
    print(f"After scaling - Sample K matrix:\n{all_Ks[0]}")
    print(f"all_c2ws shape: {all_c2ws.shape}")
    print(f"all_Ks shape: {all_Ks.shape}")

    input_indices = list(range(num_inputs))
    
    # 创建图像条件 (img2img不需要prior_indices)
    image_cond = {
        "img": all_imgs_data,  # 使用列表格式而非numpy数组
        "input_indices": input_indices,
    }
    
    # 创建相机条件
    camera_cond = {
        "c2w": all_c2ws,
        "K": all_Ks,
        "input_indices": list(range(num_inputs + num_targets)),
    }

    # 调用run_one_scene进行生成 (img2img是单阶段)
    video_path_generator = run_one_scene(
        task="img2img",
        version_dict={
            "H": H,
            "W": W,
            "T": VERSION_DICT["T"],
            "C": VERSION_DICT["C"],
            "f": VERSION_DICT["f"],
            "options": options,
        },
        model=diffusion_models["model"],
        ae=diffusion_models["ae"],
        conditioner=diffusion_models["conditioner"],
        denoiser=diffusion_models["denoiser"],
        image_cond=image_cond,
        camera_cond=camera_cond,
        save_path=save_dir,
        use_traj_prior=False,  # img2img不使用轨迹先验
        traj_prior_c2ws=None,
        traj_prior_Ks=None,
        seed=seed,
    )
    
    # 等待生成完成
    for _ in video_path_generator:
        pass

    # img2img是单阶段，返回samples-rgb路径
    video_path = os.path.join(save_dir, "samples-rgb")
    
    return video_path, save_dir

def create_progress_bar(phase, total, callback):
    """创建一个简单的进度条对象，兼容run_one_scene的进度条需求"""
    class CustomProgressBar:
        def __init__(self, phase, total, callback):
            self.phase = phase
            self.total = total
            self.current = 0
            self.callback = callback
            
        def update(self, n=1):
            self.current += n
            if self.callback:
                self.callback(self.phase, self.current, self.total)
                
        def tqdm(self, iterable=None, desc=None, total=None, **kwargs):
            return self
    
    return CustomProgressBar(phase, total, callback)

from PIL import Image
def stitch_images(folder_path, output_file=None):
    """
    将指定文件夹中的所有PNG图片按每行最多4个的方式堆叠在一起，
    并保存为一个JPG文件。
    
    参数:
    folder_path -- 包含PNG图片的文件夹路径
    output_file -- 输出的JPG文件路径，如果为None则保存在输入文件夹中
    """
    # 如果未指定输出文件路径，默认保存在输入文件夹中
    if output_file is None:
        output_file = os.path.join(folder_path, 'combined_image.jpg')
    # 获取所有PNG文件并排序
    png_files = sorted(glob.glob(os.path.join(folder_path, '*.png')))
    
    if not png_files:
        print(f"在 {folder_path} 中没有找到PNG文件。")
        return
    
    # 加载所有图片
    images = [Image.open(f) for f in png_files]
    
    # 计算每个图片的尺寸（假设所有图片尺寸相同，使用第一张图片的尺寸）
    width, height = images[0].size
    
    # 计算行数和列数
    n_images = len(images)
    cols = min(5, n_images)  # 每行最多4张图片
    rows = (n_images + cols - 1) // cols  # 向上取整计算行数
    
    # 创建新图片
    combined_width = cols * width
    combined_height = rows * height
    combined_image = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
    
    # 放置图片
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        combined_image.paste(img, (col * width, row * height))
    
    # 保存合并后的图片
    combined_image.save(output_file, 'JPEG', quality=95)
    print(f"已将 {n_images} 张PNG图片合并为 {output_file}")