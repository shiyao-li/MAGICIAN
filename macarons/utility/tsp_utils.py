import os
import random
import time
import torch
import heapq
import torchvision
import shutil
import numpy as np
import trimesh



def generate_key_value_splited_dict(orig_dict):
    '''
    Generate a splited dictionary:{position_index, real_position_camera}
    '''
    new_dict = {}

    for key, value in orig_dict.items():
        new_key = key.split(",")[0:3]
        new_key = ", ".join(new_key) + "]"
        
        new_value = value[0]
        
        new_dict[new_key] = new_value
    return new_dict

# axis_to_mirror = [0, 1, 2]
def line_segment_intersects_point_cloud_region(point_cloud, start_point, end_point):
    # point_cloud = scene
    line_vector = end_point - start_point

    # Check the line segment length to prevent division by a very small number
    line_length_squared = torch.norm(line_vector)**2
    # if line_length_squared < 1e-8:  # Use a small threshold, e.g., 1e-8
    #     return False

    # Calculate vectors from each point in the cloud to the start and end of the line segment
    point_to_start = point_cloud - start_point
    point_to_end = point_cloud - end_point

    # Compute projection vectors using vectorized operations
    projection_vector = torch.sum(point_to_start * line_vector, dim=1) / line_length_squared

    # Create a mask to identify points within the line segment range
    within_segment_mask = (projection_vector >= 0) & (projection_vector <= 1)

    # Compute closest points on the line segment for points within the segment range
    closest_point = start_point + projection_vector.unsqueeze(1) * line_vector
    distances_within_segment = torch.norm(point_cloud - closest_point, dim=1)

    # Compute the minimum distance to the line segment endpoints for points outside the segment range
    distances_outside_segment = torch.min(torch.norm(point_to_start, dim=1), torch.norm(point_to_end, dim=1))

    # Combine distances for both cases
    distances = torch.where(within_segment_mask, distances_within_segment, distances_outside_segment)

    # Check for intersection
    if distances.numel() == 0:
        return False 
    if torch.min(distances).item() < 1:  # Intersection threshold
        return True
    return False

# def generate_Dijkstra_path(pose_space, start_position, end_position, mesh, device):

#     def get_neighbors(position):
#         x, y, z = position
#         potential_neighbors = [
#             [x+1, y, z],
#             [x-1, y, z],
#             [x, y+1, z],
#             [x, y-1, z],
#             [x, y, z+1],
#             [x, y, z-1]
#         ]
#         neighbors = [n for n in potential_neighbors if str(n).replace(", ", ",  ") in pose_space 
#                     # and not line_segment_intersects_point_cloud_region(point_cloud, pose_space[str(position).replace(", ", ",  ")], pose_space[str(n).replace(", ", ",  ")])]
#                     and not line_segment_mesh_intersection(pose_space[str(position).replace(", ", ",  ")], pose_space[str(n).replace(", ", ",  ")], mesh)]
#         return neighbors

#     start = tuple(start_position)
#     goal = tuple(end_position)

#     frontier = []  # Using a priority queue with (cost, position)
#     heapq.heappush(frontier, (0, start))
    
#     came_from = {start: None}
#     cost_so_far = {start: 0}

#     while frontier:
#         current_cost, current = heapq.heappop(frontier)

#         if current == goal:
#             break

#         for next in get_neighbors(list(current)):
#             next_tuple = tuple(next)
#             new_cost = cost_so_far[current] + 1  # Since every step has a cost of 1
#             if next_tuple not in cost_so_far or new_cost < cost_so_far[next_tuple]:
#                 cost_so_far[next_tuple] = new_cost
#                 heapq.heappush(frontier, (new_cost, next_tuple))
#                 came_from[next_tuple] = current

#     # reconstruct the path
#     if goal in came_from:  # ensure there is a path
#         path = []
#         real_move_path = []
#         current = goal
#         while current:
#             path.append(list(current))
#             current = came_from[current]
#         path.reverse()
#         for idx in path:
#             real_move_path.append(torch.tensor(idx).to(device))
#         return real_move_path[1:]
#     else:
#         return None


# def line_segment_mesh_intersection(start_point, end_point, mesh):
#     """
#     Compute the intersection between a line segment and a mesh.
#     start_point: (torch.Tensor) Shape (3,)
#     end_point: (torch.Tensor) Shape (3,)
#     mesh: trimesh object
#     """
#     if not isinstance(start_point, list):
#         start_point = start_point.cpu().numpy()
#         end_point = end_point.cpu().numpy()
#     else:
#         start_point = np.array(start_point)
#         end_point = np.array(end_point)
#     direction = end_point - start_point
#     line_length = np.linalg.norm(direction)
#     direction = direction / line_length

#     ray_origins = np.array([start_point])
#     ray_directions = np.array([direction])

#     locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=ray_origins,
#                                                                    ray_directions=ray_directions)
#     if locations.size > 0:
#         distances = np.linalg.norm(locations - start_point, axis=1)
#         intersection_on_segment = any(distances < line_length)
        
#         if intersection_on_segment:
#             return True
#         else:
#             return False
#     else:
#         return False
def line_segment_mesh_intersection(start_point, end_point, mesh):
    """优化版本：一旦发现碰撞就立即返回"""
    # if not isinstance(start_point, list):
    #     start_point = start_point.cpu().numpy()
    #     end_point = end_point.cpu().numpy()
    # else:
    #     start_point = np.array(start_point)
    #     end_point = np.array(end_point)
    
    direction = end_point - start_point
    line_length = np.linalg.norm(direction)
    
    if line_length < 1e-6:  # 添加零长度检查
        return False
    
    direction = direction / line_length
    
    locations, _, _ = mesh.intersects_location(
        ray_origins=[start_point],
        ray_directions=[direction]
    )
    
    if len(locations) == 0:
        return False
    
    # 向量化计算所有距离，使用any()短路
    distances = np.linalg.norm(locations - start_point, axis=1)
    return np.any(distances < line_length)

# def solve_tsp_greedy(start_point, points_list, pose_space, mesh, device):
#     """
#     使用贪心算法求解TSP问题
    
#     Args:
#         start_point: (x, y, z) 起点坐标
#         points_list: [(x1,y1,z1), (x2,y2,z2), ...] 需要访问的点列表
#         pose_space: 位姿空间
#         mesh: 网格
#         device: 设备
    
#     Returns:
#         (总距离, 访问顺序, 完整轨迹, 跳过的点)
#     """
#     if not points_list:
#         return 0, [start_point], [start_point], []
    
#     current_pos = start_point
#     unvisited = set(range(len(points_list)))
#     visit_order = [start_point]
#     full_trajectory = [start_point]
#     total_distance = 0
    
#     # 访问所有点
#     while unvisited:
#         print(f"当前位置: {current_pos}, 剩余点数: {len(unvisited)}")
        
#         # 找到距离当前位置最近的未访问点
#         min_dist = float('inf')
#         next_idx = -1
#         best_trajectory = None
        
#         for idx in unvisited:
#             trajectory = generate_Dijkstra_path(pose_space, current_pos, points_list[idx], mesh, device)
#             if trajectory is None:
#                 print(f"警告：无法到达点 {points_list[idx]}，跳过")
#                 continue
            
#             dist = len(trajectory)
#             if dist < min_dist:
#                 min_dist = dist
#                 next_idx = idx
#                 best_trajectory = trajectory
        
#         # 检查是否找到了可达的点
#         if next_idx == -1:
#             print(f"警告：剩余 {len(unvisited)} 个点无法到达，结束搜索")
#             break
        
#         # 移动到最近的点
#         next_point = points_list[next_idx]
#         trajectory = best_trajectory
        
#         # 更新状态
#         unvisited.remove(next_idx)
#         visit_order.append(next_point)
#         full_trajectory.extend(trajectory[1:])  # 跳过起点避免重复
#         total_distance += min_dist
#         current_pos = next_point
#         print(f"访问点 {next_point}，距离: {min_dist}")
    
#     # 返回未访问的点
#     skipped_points = [points_list[i] for i in unvisited]
#     print(len(full_trajectory))
#     print(full_trajectory)
#     return total_distance, visit_order, full_trajectory, skipped_points

import heapq
import torch

def generate_Dijkstra_path(pose_space, start_position, end_position, mesh, device):
    """
    生成基于位置的Dijkstra路径，然后添加相机角度变化
    
    Args:
        pose_space: 位姿空间字典
        start_position: [x, y, z, e, a] 起始5D索引
        end_position: [x, y, z, e, a] 终点5D索引
        mesh: 网格
        device: 设备
    
    Returns:
        完整的5D轨迹列表，如果无法到达返回None
    """
    
    def get_neighbors(position):
        x, y, z = position
        potential_neighbors = [
            [x+1, y, z],
            [x-1, y, z],
            [x, y+1, z],
            [x, y-1, z],
            [x, y, z+1],
            [x, y, z-1]
        ]
        neighbors = [n for n in potential_neighbors if str(n).replace(", ", ",  ") in pose_space 
                    and not line_segment_mesh_intersection(pose_space[str(position).replace(", ", ",  ")].cpu().numpy(), pose_space[str(n).replace(", ", ",  ")].cpu().numpy(), mesh)]
        return neighbors

    # 提取3D位置进行路径规划
    start_3d = tuple(start_position[:3])
    end_3d = tuple(end_position[:3])
    
    # 如果起点和终点位置相同，只需要处理角度变化
    if start_3d == end_3d:
        return interpolate_camera_angles(start_position, end_position, device)

    # 使用Dijkstra算法规划3D路径
    frontier = []
    heapq.heappush(frontier, (0, start_3d))
    
    came_from = {start_3d: None}
    cost_so_far = {start_3d: 0}

    while frontier:
        current_cost, current = heapq.heappop(frontier)

        if current == end_3d:
            break

        for next in get_neighbors(list(current)):
            next_tuple = tuple(next)
            new_cost = cost_so_far[current] + 1
            if next_tuple not in cost_so_far or new_cost < cost_so_far[next_tuple]:
                cost_so_far[next_tuple] = new_cost
                heapq.heappush(frontier, (new_cost, next_tuple))
                came_from[next_tuple] = current

    # 重构3D路径
    if end_3d not in came_from:
        return None
    
    path_3d = []
    current = end_3d
    while current:
        path_3d.append(list(current))
        current = came_from[current]
    path_3d.reverse()
    
    # # 生成完整的5D轨迹
    # full_trajectory = generate_5d_trajectory(path_3d, start_position, end_position, device)
    
    return path_3d[1:]


def interpolate_camera_angles(start_pose, end_pose, device):
    """
    在相同位置上插值相机角度
    
    Args:
        start_pose: [x, y, z, e, a] 起始5D索引
        end_pose: [x, y, z, e, a] 终点5D索引
        device: 设备
    
    Returns:
        角度插值轨迹
    """
    trajectory = []
    current = start_pose.copy()
    
    # 插值elevation
    while current[3] != end_pose[3]:
        if current[3] < end_pose[3]:
            current[3] += 1
        else:
            current[3] -= 1
        trajectory.append(torch.tensor(current.copy()).to(device))
    
    # 插值azimuth（考虑循环）
    while current[4] != end_pose[4]:
        # 计算最短路径方向
        diff = (end_pose[4] - current[4]) % 10
        if diff <= 5:
            current[4] = (current[4] + 1) % 10
        else:
            current[4] = (current[4] - 1) % 10
        trajectory.append(torch.tensor(current.copy()).to(device))
    
    return trajectory


def generate_5d_trajectory(path_3d, start_pose, end_pose, device):
    """
    基于3D路径生成5D轨迹，角度调整和位置移动交替进行
    
    Args:
        path_3d: 3D位置路径
        start_pose: [x, y, z, e, a] 起始5D索引
        end_pose: [x, y, z, e, a] 终点5D索引
        device: 设备
    
    Returns:
        完整的5D轨迹
    """
    if len(path_3d) <= 1:
        return interpolate_camera_angles(start_pose, end_pose, device)
    
    trajectory = []
    current_pose = start_pose.copy()
    
    # 计算总的位置步数和角度步数
    total_position_steps = len(path_3d) - 1
    total_angle_steps = calculate_angle_steps(start_pose, end_pose)
    
    # 如果没有角度变化，直接处理位置移动
    if total_angle_steps == 0:
        for i in range(1, len(path_3d)):
            current_pose[:3] = path_3d[i]
            trajectory.append(torch.tensor(current_pose.copy()).to(device))
        return trajectory
    
    # 交替进行位置移动和角度调整
    position_index = 1
    angle_progress = [0, 0]  # [elevation_progress, azimuth_progress]
    
    while position_index < len(path_3d) or not is_angle_complete(current_pose, end_pose):
        # 决定这一步是移动位置还是调整角度
        if should_move_position(position_index, len(path_3d), angle_progress, total_angle_steps):
            # 移动位置
            current_pose[:3] = path_3d[position_index]
            trajectory.append(torch.tensor(current_pose.copy()).to(device))
            position_index += 1
        else:
            # 调整角度
            current_pose, angle_progress = adjust_one_angle_step(current_pose, end_pose, angle_progress)
            trajectory.append(torch.tensor(current_pose.copy()).to(device))
    
    return trajectory


def calculate_angle_steps(start_pose, end_pose):
    """计算角度调整需要的总步数"""
    elevation_steps = abs(end_pose[3] - start_pose[3])
    
    # 计算azimuth的最短路径步数
    azimuth_diff = (end_pose[4] - start_pose[4]) % 10
    azimuth_steps = min(azimuth_diff, 10 - azimuth_diff)
    
    return elevation_steps + azimuth_steps


def is_angle_complete(current_pose, end_pose):
    """检查角度是否已经调整完成"""
    return current_pose[3] == end_pose[3] and current_pose[4] == end_pose[4]


def should_move_position(position_index, total_positions, angle_progress, total_angle_steps):
    """决定这一步是移动位置还是调整角度"""
    if position_index >= total_positions:
        return False  # 位置已经全部移动完成
    
    if sum(angle_progress) >= total_angle_steps:
        return True  # 角度已经调整完成，继续移动位置
    
    # 交替策略：尽量让位置移动和角度调整平衡进行
    position_remaining = total_positions - position_index
    angle_remaining = total_angle_steps - sum(angle_progress)
    
    if position_remaining == 0:
        return False
    if angle_remaining == 0:
        return True
    
    # 简单的交替策略：如果位置进度落后于角度进度，就移动位置
    position_progress = (position_index - 1) / (total_positions - 1) if total_positions > 1 else 1
    angle_progress_ratio = sum(angle_progress) / total_angle_steps if total_angle_steps > 0 else 1
    
    return position_progress <= angle_progress_ratio


def adjust_one_angle_step(current_pose, end_pose, angle_progress):
    """执行一步角度调整"""
    new_pose = current_pose.copy()
    new_progress = angle_progress.copy()
    
    # 优先调整elevation
    if new_pose[3] != end_pose[3]:
        if new_pose[3] < end_pose[3]:
            new_pose[3] += 1
        else:
            new_pose[3] -= 1
        new_progress[0] += 1
    # 然后调整azimuth
    elif new_pose[4] != end_pose[4]:
        # 计算最短路径方向
        diff = (end_pose[4] - new_pose[4]) % 10
        if diff <= 5:
            new_pose[4] = (new_pose[4] + 1) % 10
        else:
            new_pose[4] = (new_pose[4] - 1) % 10
        new_progress[1] += 1
    
    return new_pose, new_progress


def solve_tsp_greedy(start_point, points_list, pose_space, mesh, device):
    """
    使用贪心算法求解TSP问题
    
    Args:
        start_point: [x, y, z, e, a] 起点5D索引
        points_list: [[x1,y1,z1,e1,a1], ...] 需要访问的5D索引点列表
        pose_space: 位姿空间
        mesh: 网格
        device: 设备
    
    Returns:
        (总距离, 访问顺序, 完整轨迹, 跳过的点)
    """
    if not points_list:
        return 0, [start_point], [start_point], []
    
    current_pos = start_point
    unvisited = set(range(len(points_list)))
    visit_order = [start_point]
    full_trajectory = [torch.tensor(start_point).to(device)]
    total_distance = 0
    
    # 访问所有点
    while unvisited:
        print(f"当前位置: {current_pos}, 剩余点数: {len(unvisited)}")
        
        # 找到距离当前位置最近的未访问点
        min_dist = float('inf')
        next_idx = -1
        best_trajectory = None
        
        for idx in unvisited:
            trajectory = generate_Dijkstra_path(pose_space, current_pos, points_list[idx], mesh, device)
            if trajectory is None:
                print(f"警告：无法到达点 {points_list[idx]}，跳过")
                continue
            
            dist = len(trajectory)
            if dist < min_dist:
                min_dist = dist
                next_idx = idx
                best_trajectory = trajectory
        
        # 检查是否找到了可达的点
        if next_idx == -1:
            print(f"警告：剩余 {len(unvisited)} 个点无法到达，结束搜索")
            break
        
        # 移动到最近的点
        next_point = points_list[next_idx]
        trajectory = best_trajectory
        
        # 更新状态
        unvisited.remove(next_idx)
        visit_order.append(next_point)
        full_trajectory.extend(trajectory)  # 直接添加轨迹
        total_distance += min_dist
        current_pos = next_point
        print(f"访问点 {next_point}，距离: {min_dist}")
    
    # 返回未访问的点
    skipped_points = [points_list[i] for i in unvisited]
    print(f"完整轨迹长度: {len(full_trajectory)}")
    print(full_trajectory)
    return total_distance, visit_order, full_trajectory, skipped_points