# import torch
# import numpy as np
# import random
# import math
# from collections import defaultdict

# class MCTSNode:
#     def __init__(self, pose_key, path, remaining_steps, coverage_mask, parent=None):
#         self.pose_key = pose_key
#         self.path = path  # 完整路径序列
#         self.remaining_steps = remaining_steps
#         self.coverage_mask = coverage_mask
#         self.parent = parent
        
#         self.visit_count = 0
#         self.total_reward = 0.0
#         self.children = {}  # action(next_pose_key) -> child_node
        
#     def is_fully_expanded(self, valid_actions):
#         return len(self.children) == len(valid_actions)
    
#     def best_child(self, c=1.4):
#         """选择UCB值最高的子节点"""
#         choices = []
#         for action, child in self.children.items():
#             if child.visit_count == 0:
#                 ucb_value = float('inf')
#             else:
#                 q_value = child.total_reward / child.visit_count
#                 ucb_value = q_value + c * math.sqrt(2 * math.log(self.visit_count) / child.visit_count)
#             choices.append((ucb_value, action, child))
        
#         return max(choices, key=lambda x: x[0])[2]
    
#     def add_child(self, action, child_node):
#         self.children[action] = child_node

# class MCTSPathPlanner:
#     def __init__(self, anchor_poses, pose_coverage_cache, device='cuda'):
#         """
#         Args:
#             anchor_poses: List[pose_key] - 所有anchor poses
#             pose_coverage_cache: Dict[pose_key -> torch.tensor] - 每个pose的覆盖mask
#         """
#         self.anchor_poses = set(anchor_poses)
#         self.pose_coverage_cache = pose_coverage_cache
#         self.device = device
        
#         # 预计算每个pose的邻居
#         self.neighbors_cache = self._build_neighbors_cache()
        
#         # 计算最大可能覆盖用于奖励归一化
#         total_coverage = torch.zeros_like(next(iter(pose_coverage_cache.values())), device=device)
#         for mask in pose_coverage_cache.values():
#             total_coverage = total_coverage | mask
#         self.max_coverage = total_coverage.sum().item()
    
#     def _parse_pose_key(self, pose_key):
#         """解析pose_key为索引列表"""
#         import ast
#         return ast.literal_eval(pose_key)
    
#     def _index_distance(self, indices1, indices2):
#         """计算两个pose索引之间的步数距离（考虑环形拓扑）"""
#         max_values = [8, 4, 6, 5, 10]
        
#         total_distance = 0
#         for i, (a, b) in enumerate(zip(indices1, indices2)):
#             if i < len(max_values):
#                 max_val = max_values[i]
#                 direct_dist = abs(a - b)
#                 wrap_dist = max_val - direct_dist
#                 min_dist = min(direct_dist, wrap_dist)
#                 total_distance += min_dist
#             else:
#                 total_distance += abs(a - b)
        
#         return total_distance
    
#     def _build_neighbors_cache(self, step_limit=3):
#         """预计算每个anchor pose的邻居"""
#         print("Building neighbors cache...")
#         neighbors_cache = {}
        
#         pose_indices = {}
#         for pose_key in self.anchor_poses:
#             pose_indices[pose_key] = self._parse_pose_key(pose_key)
        
#         for pose_key, indices in pose_indices.items():
#             neighbors = []
#             for other_key, other_indices in pose_indices.items():
#                 if pose_key != other_key:
#                     distance = self._index_distance(indices, other_indices)
#                     if distance <= step_limit:
#                         neighbors.append((other_key, distance))
#             neighbors_cache[pose_key] = neighbors
            
#         print(f"Neighbors cache built for {len(neighbors_cache)} poses")
#         return neighbors_cache
    
#     def _get_valid_actions(self, current_pose, remaining_steps):
#         """获取当前pose的有效动作"""
#         if current_pose not in self.neighbors_cache:
#             return []
        
#         valid_actions = []
#         for neighbor_key, distance in self.neighbors_cache[current_pose]:
#             if distance <= remaining_steps:
#                 valid_actions.append((neighbor_key, distance))
        
#         return valid_actions
    
#     def _calculate_path_reward(self, path):
#         """计算路径的总覆盖奖励"""
#         if not path:
#             return 0.0
            
#         total_coverage = torch.zeros_like(next(iter(self.pose_coverage_cache.values())), device=self.device)
#         for pose_key in path:
#             if pose_key in self.pose_coverage_cache:
#                 total_coverage = total_coverage | self.pose_coverage_cache[pose_key]
        
#         coverage_count = total_coverage.sum().item()
#         return coverage_count / self.max_coverage  # 归一化到[0,1]
    
#     def _simulate_random_path(self, node):
#         """从给定节点开始随机模拟路径"""
#         current_pose = node.pose_key
#         remaining_steps = node.remaining_steps
#         path = node.path.copy()
        
#         while remaining_steps > 0:
#             valid_actions = self._get_valid_actions(current_pose, remaining_steps)
#             if not valid_actions:
#                 break
                
#             # 随机选择一个有效动作
#             next_pose, step_cost = random.choice(valid_actions)
#             path.append(next_pose)
#             current_pose = next_pose
#             remaining_steps -= step_cost
        
#         return self._calculate_path_reward(path)
    
#     def _select(self, root):
#         """Selection: 选择到叶子节点"""
#         node = root
#         while node.children and node.is_fully_expanded(self._get_valid_actions(node.pose_key, node.remaining_steps)):
#             node = node.best_child()
#         return node
    
#     def _expand(self, node):
#         """Expansion: 展开一个新子节点"""
#         valid_actions = self._get_valid_actions(node.pose_key, node.remaining_steps)
        
#         if not valid_actions:
#             return node
        
#         # 找到未展开的动作
#         unexplored_actions = []
#         for next_pose, step_cost in valid_actions:
#             if next_pose not in node.children:
#                 unexplored_actions.append((next_pose, step_cost))
        
#         if not unexplored_actions:
#             return node
        
#         # 随机选择一个未展开的动作
#         next_pose, step_cost = random.choice(unexplored_actions)
        
#         # 创建子节点
#         new_path = node.path + [next_pose]
#         new_remaining = node.remaining_steps - step_cost
        
#         # 计算新的覆盖mask
#         new_coverage = node.coverage_mask.clone()
#         if next_pose in self.pose_coverage_cache:
#             new_coverage = new_coverage | self.pose_coverage_cache[next_pose]
        
#         child_node = MCTSNode(next_pose, new_path, new_remaining, new_coverage, parent=node)
#         node.add_child(next_pose, child_node)
        
#         return child_node
    
#     def _backpropagate(self, node, reward):
#         """Backpropagation: 向上传播奖励"""
#         while node is not None:
#             node.visit_count += 1
#             node.total_reward += reward
#             node = node.parent
    
#     def plan_optimal_path(self, start_pose_key, max_steps, simulation_count=1000):
#         """
#         使用MCTS规划最优路径
        
#         Args:
#             start_pose_key: 起始pose
#             max_steps: 最大步数预算
#             simulation_count: MCTS模拟次数
            
#         Returns:
#             List[pose_key]: 最优路径
#             float: 路径覆盖奖励
#         """
#         print(f"Starting MCTS planning from {start_pose_key}, max_steps={max_steps}")
        
#         # 初始化根节点
#         if start_pose_key not in self.pose_coverage_cache:
#             raise ValueError(f"Start pose {start_pose_key} not in coverage cache")
        
#         initial_coverage = self.pose_coverage_cache[start_pose_key].clone()
#         root = MCTSNode(start_pose_key, [start_pose_key], max_steps, initial_coverage)
        
#         # MCTS主循环
#         for i in range(simulation_count):
#             if (i + 1) % 100 == 0:
#                 print(f"MCTS simulation {i + 1}/{simulation_count}")
            
#             # 1. Selection
#             leaf_node = self._select(root)
            
#             # 2. Expansion
#             new_node = self._expand(leaf_node)
            
#             # 3. Simulation
#             reward = self._simulate_random_path(new_node)
            
#             # 4. Backpropagation
#             self._backpropagate(new_node, reward)
        
#         # 提取最优路径
#         best_path = self._extract_best_path(root)
#         best_reward = self._calculate_path_reward(best_path)
        
#         print(f"MCTS completed. Best path length: {len(best_path)}")
#         print(f"Best path coverage: {best_reward * self.max_coverage:.0f}/{self.max_coverage} ({best_reward*100:.1f}%)")
        
#         return best_path, best_reward
    
#     def _extract_best_path(self, root):
#         """提取访问次数最高的路径"""
#         path = [root.pose_key]
#         node = root
        
#         while node.children:
#             # 选择访问次数最多的子节点
#             best_child = max(node.children.values(), key=lambda x: x.visit_count)
#             path.append(best_child.pose_key)
#             node = best_child
        
#         return path

# # 使用示例
# def plan_path_with_mcts(anchor_poses, pose_coverage_cache, start_pose, max_steps, simulation_count=1000):
#     """
#     便捷函数：使用MCTS规划路径
    
#     Args:
#         anchor_poses: select_dense_graph_poses返回的anchor poses列表
#         pose_coverage_cache: select_dense_graph_poses返回的覆盖缓存
#         start_pose: 起始pose key
#         max_steps: 最大步数
#         simulation_count: MCTS模拟次数
        
#     Returns:
#         best_path: 最优路径
#         coverage_ratio: 覆盖比例
#     """
#     planner = MCTSPathPlanner(anchor_poses, pose_coverage_cache)
#     best_path, coverage_ratio = planner.plan_optimal_path(start_pose, max_steps, simulation_count)
    
#     return best_path, coverage_ratio


import torch
import numpy as np
import random
import math
from collections import defaultdict

# class MCTSNode:
#     def __init__(self, pose_key, path, remaining_steps, coverage_mask, parent=None):
#         self.pose_key = pose_key
#         self.path = path  # anchor poses路径
#         self.remaining_steps = remaining_steps
#         self.coverage_mask = coverage_mask
#         self.parent = parent
        
#         self.visit_count = 0
#         self.total_reward = 0.0
#         self.children = {}
        
#     def is_fully_expanded(self, valid_actions):
#         return len(self.children) == len(valid_actions)
    
#     def best_child(self, c=1.4):
#         choices = []
#         for action, child in self.children.items():
#             if child.visit_count == 0:
#                 ucb_value = float('inf')
#             else:
#                 q_value = child.total_reward / child.visit_count
#                 ucb_value = q_value + c * math.sqrt(2 * math.log(self.visit_count) / child.visit_count)
#             choices.append((ucb_value, action, child))
#         return max(choices, key=lambda x: x[0])[2]
    
#     def add_child(self, action, child_node):
#         self.children[action] = child_node

# class MCTSPathPlanner:
#     def __init__(self, anchor_poses, pose_coverage_cache, device='cuda'):
#         self.anchor_poses = set(anchor_poses)
#         self.pose_coverage_cache = pose_coverage_cache
#         self.device = device
#         self.neighbors_cache = {}
        
#         # 计算最大覆盖
#         total_coverage = torch.zeros_like(next(iter(pose_coverage_cache.values())), device=device)
#         for mask in pose_coverage_cache.values():
#             total_coverage = total_coverage | mask
#         self.max_coverage = total_coverage.sum().item()
    
#     def _parse_pose_key(self, pose_key):
#         import ast
#         return ast.literal_eval(pose_key)
    
#     def _index_distance(self, indices1, indices2):
#         """计算两个pose索引之间的步数距离（部分维度考虑环形拓扑）"""
#         max_values = [8, 4, 6, 5, 10]
#         is_circular = [False, False, False, True, True]  # 只有后两个维度是环形的
        
#         total_distance = 0
#         for i, (a, b) in enumerate(zip(indices1, indices2)):
#             if i < len(max_values):
#                 max_val = max_values[i]
#                 if is_circular[i]:
#                     # 环形距离：考虑wrap-around
#                     direct_dist = abs(a - b)
#                     wrap_dist = max_val - direct_dist
#                     min_dist = min(direct_dist, wrap_dist)
#                     total_distance += min_dist
#                 else:
#                     # 普通线性距离
#                     total_distance += abs(a - b)
#             else:
#                 # 如果有额外维度，使用普通距离
#                 total_distance += abs(a - b)
        
#         return total_distance

    
#     def _build_neighbors_cache(self, start_pose_key, step_limit=3):
#         all_poses = set(self.anchor_poses)
#         all_poses.add(start_pose_key)
        
#         pose_indices = {}
#         for pose_key in all_poses:
#             pose_indices[pose_key] = self._parse_pose_key(pose_key)
        
#         neighbors_cache = {}
#         for pose_key, indices in pose_indices.items():
#             neighbors = []
#             for other_key in self.anchor_poses:
#                 if pose_key != other_key:
#                     other_indices = pose_indices[other_key]
#                     distance = self._index_distance(indices, other_indices)
#                     if distance <= step_limit:
#                         neighbors.append((other_key, distance))
#             neighbors_cache[pose_key] = neighbors
#         return neighbors_cache
    
#     def _get_valid_actions(self, current_pose, remaining_steps):
#         if current_pose not in self.neighbors_cache:
#             return []
#         valid_actions = []
#         for neighbor_key, distance in self.neighbors_cache[current_pose]:
#             if distance <= remaining_steps:
#                 valid_actions.append((neighbor_key, distance))
#         return valid_actions
    
#     def _calculate_anchor_reward(self, anchor_path):
#         if not anchor_path:
#             return 0.0
#         total_coverage = torch.zeros_like(next(iter(self.pose_coverage_cache.values())), device=self.device)
#         for pose_key in anchor_path:
#             if pose_key in self.pose_coverage_cache:
#                 total_coverage = total_coverage | self.pose_coverage_cache[pose_key]
#         coverage_count = total_coverage.sum().item()
#         return coverage_count / self.max_coverage
    
#     def _calculate_anchor_steps(self, anchor_path):
#         if len(anchor_path) <= 1:
#             return 0
#         total_steps = 0
#         for i in range(len(anchor_path) - 1):
#             current_indices = self._parse_pose_key(anchor_path[i])
#             next_indices = self._parse_pose_key(anchor_path[i + 1])
#             total_steps += self._index_distance(current_indices, next_indices)
#         return total_steps
    
#     def _generate_step_by_step_path(self, start_indices, end_indices):
#         """生成逐步路径，部分维度考虑环形拓扑"""
#         max_values = [8, 4, 6, 5, 10]
#         is_circular = [False, False, False, True, True]  # 只有后两个维度是环形的
#         current = start_indices.copy()
#         path = [str(current)]
        
#         while current != end_indices:
#             # 找到需要改变的维度
#             for dim in range(len(current)):
#                 if current[dim] != end_indices[dim]:
#                     if dim < len(max_values):
#                         max_val = max_values[dim]
#                         if is_circular[dim]:
#                             # 环形维度：选择最短环形路径方向
#                             diff = end_indices[dim] - current[dim]
#                             if abs(diff) <= max_val // 2:
#                                 step = 1 if diff > 0 else -1
#                             else:
#                                 step = -1 if diff > 0 else 1
#                             current[dim] = (current[dim] + step) % max_val
#                         else:
#                             # 线性维度：直接步进，但需要边界检查
#                             step = 1 if end_indices[dim] > current[dim] else -1
#                             new_val = current[dim] + step
#                             if 0 <= new_val < max_val:  # 边界检查
#                                 current[dim] = new_val
#                             else:
#                                 # 如果超出边界，跳过此维度
#                                 continue
#                     else:
#                         # 超出预定义范围的维度，使用普通步进
#                         step = 1 if end_indices[dim] > current[dim] else -1
#                         current[dim] += step
#                     path.append(str(current))
#                     break
#         return path
    
#     def _expand_to_full_path(self, anchor_path):
#         if len(anchor_path) <= 1:
#             return anchor_path
        
#         full_path = [anchor_path[0]]
#         for i in range(len(anchor_path) - 1):
#             start_indices = self._parse_pose_key(anchor_path[i])
#             end_indices = self._parse_pose_key(anchor_path[i + 1])
#             step_path = self._generate_step_by_step_path(start_indices, end_indices)
#             full_path.extend(step_path[1:])  # 跳过起始点
#         return full_path
    
#     def _calculate_full_path_reward(self, full_path):
#         if not full_path:
#             return 0.0
#         total_coverage = torch.zeros_like(next(iter(self.pose_coverage_cache.values())), device=self.device)
#         for pose_key in full_path:
#             if pose_key in self.pose_coverage_cache:
#                 total_coverage = total_coverage | self.pose_coverage_cache[pose_key]
#         coverage_count = total_coverage.sum().item()
#         return coverage_count / self.max_coverage
    
#     def _simulate_random_path(self, node):
#         current_pose = node.pose_key
#         remaining_steps = node.remaining_steps
#         anchor_path = node.path.copy()
        
#         while remaining_steps > 0:
#             valid_actions = self._get_valid_actions(current_pose, remaining_steps)
#             if not valid_actions:
#                 break
#             next_anchor, step_cost = random.choice(valid_actions)
#             anchor_path.append(next_anchor)
#             current_pose = next_anchor
#             remaining_steps -= step_cost
        
#         # 计算anchor路径奖励
#         reward = self._calculate_anchor_reward(anchor_path)
        
#         # 跟踪最优路径
#         anchor_steps = self._calculate_anchor_steps(anchor_path)
#         if not hasattr(self, 'best_anchor_path') or reward > self.best_anchor_reward:
#             self.best_anchor_path = anchor_path
#             self.best_anchor_reward = reward
#             self.best_anchor_steps = anchor_steps
        
#         return reward
    
#     def _select(self, root):
#         node = root
#         while node.children and node.is_fully_expanded(self._get_valid_actions(node.pose_key, node.remaining_steps)):
#             node = node.best_child()
#         return node
    
#     def _expand(self, node):
#         valid_actions = self._get_valid_actions(node.pose_key, node.remaining_steps)
#         if not valid_actions:
#             return node
        
#         unexplored_actions = []
#         for next_pose, step_cost in valid_actions:
#             if next_pose not in node.children:
#                 unexplored_actions.append((next_pose, step_cost))
#         if not unexplored_actions:
#             return node
        
#         next_pose, step_cost = random.choice(unexplored_actions)
#         new_path = node.path + [next_pose]
#         new_remaining = node.remaining_steps - step_cost
        
#         new_coverage = node.coverage_mask.clone()
#         if next_pose in self.pose_coverage_cache:
#             new_coverage = new_coverage | self.pose_coverage_cache[next_pose]
        
#         child_node = MCTSNode(next_pose, new_path, new_remaining, new_coverage, parent=node)
#         node.add_child(next_pose, child_node)
#         return child_node
    
#     def _backpropagate(self, node, reward):
#         while node is not None:
#             node.visit_count += 1
#             node.total_reward += reward
#             node = node.parent
    
#     def plan_optimal_path(self, start_pose_key, max_steps, simulation_count=1000):
#         print(f"Starting MCTS planning from {start_pose_key}, max_steps={max_steps}")
        
#         self.neighbors_cache = self._build_neighbors_cache(start_pose_key)
        
#         if start_pose_key not in self.neighbors_cache or not self.neighbors_cache[start_pose_key]:
#             return {'anchor_path': [start_pose_key], 'full_path': [start_pose_key], 'anchor_steps': 0, 'coverage': 0.0}
        
#         # 初始化
#         if start_pose_key in self.pose_coverage_cache:
#             initial_coverage = self.pose_coverage_cache[start_pose_key].clone()
#         else:
#             sample_mask = next(iter(self.pose_coverage_cache.values()))
#             initial_coverage = torch.zeros_like(sample_mask, device=self.device)
        
#         root = MCTSNode(start_pose_key, [start_pose_key], max_steps, initial_coverage)
        
#         # 初始化最优路径跟踪
#         self.best_anchor_path = [start_pose_key]
#         self.best_anchor_reward = self._calculate_anchor_reward([start_pose_key])
#         self.best_anchor_steps = 0
        
#         # MCTS主循环
#         for i in range(simulation_count):
#             if (i + 1) % 1000 == 0:
#                 current_best_coverage = self.best_anchor_reward * self.max_coverage
#                 print(f"MCTS simulation {i + 1}/{simulation_count}, best coverage: {current_best_coverage:.0f}/{self.max_coverage} ({self.best_anchor_reward*100:.1f}%)")               
            
#             leaf_node = self._select(root)
#             new_node = self._expand(leaf_node)
#             reward = self._simulate_random_path(new_node)
#             self._backpropagate(new_node, reward)
        
#         # 生成完整路径
#         full_path = self._expand_to_full_path(self.best_anchor_path)
#         full_reward = self._calculate_full_path_reward(full_path)
        
#         print(f"MCTS completed:")
#         print(f"  Anchor path: {len(self.best_anchor_path)} poses, {self.best_anchor_steps} steps")
#         print(f"  Full path: {len(full_path)} poses")
#         print(f"  Coverage: {full_reward * self.max_coverage:.0f}/{self.max_coverage} ({full_reward*100:.1f}%)")
        
#         return {
#             'anchor_path': self.best_anchor_path,
#             'full_path': full_path,
#             'anchor_steps': self.best_anchor_steps,
#             'coverage': full_reward
#         }

# def plan_path_with_mcts(anchor_poses, pose_coverage_cache, start_pose, max_steps, simulation_count=1000, device='cuda'):
#     planner = MCTSPathPlanner(anchor_poses, pose_coverage_cache, device)
#     result = planner.plan_optimal_path(start_pose, max_steps, simulation_count)
#     return result

#######################################################use max coverage / ignore middle poses
'''
import torch
import numpy as np
import random
import math
from collections import defaultdict

class MCTSNode:
    def __init__(self, pose_key, path, remaining_steps, coverage_mask, parent=None):
        self.pose_key = pose_key
        self.path = path  # anchor poses路径
        self.remaining_steps = remaining_steps
        self.coverage_mask = coverage_mask
        self.parent = parent
        
        self.visit_count = 0
        self.total_reward = 0.0
        self.children = {}
        
    def is_fully_expanded(self, valid_actions):
        return len(self.children) == len(valid_actions)
    
    def best_child(self, c=1.4, max_coverage=None):
        choices = []
        for action, child in self.children.items():
            if child.visit_count == 0:
                ucb_value = float('inf')
            else:
                # 归一化q_value
                q_value = child.total_reward / child.visit_count
                if max_coverage is not None and max_coverage > 0:
                    q_value = q_value / max_coverage
                
                ucb_value = q_value + c * math.sqrt(2 * math.log(self.visit_count) / child.visit_count)
            choices.append((ucb_value, action, child))
        return max(choices, key=lambda x: x[0])[2]
    
    def add_child(self, action, child_node):
        self.children[action] = child_node

class MCTSPathPlanner:
    def __init__(self, anchor_poses, pose_coverage_cache, device='cuda'):
        self.anchor_poses = set(anchor_poses)
        self.pose_coverage_cache = pose_coverage_cache
        self.device = device
        self.neighbors_cache = {}
        
        # 计算anchor poses的总覆盖（不包括中间poses）
        anchor_coverage = torch.zeros_like(next(iter(pose_coverage_cache.values())), device=device)
        for pose_key in anchor_poses:
            if pose_key in pose_coverage_cache:
                anchor_coverage = anchor_coverage | pose_coverage_cache[pose_key]
        self.max_anchor_coverage = anchor_coverage.sum().item()
        
        # 跟踪当前simulation阶段的最大coverage（用于UCB归一化）
        self.current_max_coverage = 0.0
        
        print(f"Maximum possible anchor coverage: {self.max_anchor_coverage} points")
    
    def _parse_pose_key(self, pose_key):
        import ast
        return ast.literal_eval(pose_key)
    
    def _index_distance(self, indices1, indices2):
        """计算两个pose索引之间的步数距离（部分维度考虑环形拓扑）"""
        max_values = [8, 4, 6, 5, 10]
        is_circular = [False, False, False, True, True]  # 只有后两个维度是环形的
        
        total_distance = 0
        for i, (a, b) in enumerate(zip(indices1, indices2)):
            if i < len(max_values):
                max_val = max_values[i]
                if is_circular[i]:
                    # 环形距离：考虑wrap-around
                    direct_dist = abs(a - b)
                    wrap_dist = max_val - direct_dist
                    min_dist = min(direct_dist, wrap_dist)
                    total_distance += min_dist
                else:
                    # 普通线性距离
                    total_distance += abs(a - b)
            else:
                # 如果有额外维度，使用普通距离
                total_distance += abs(a - b)
        
        return total_distance
    
    def _build_neighbors_cache(self, start_pose_key, step_limit=3):
        all_poses = set(self.anchor_poses)
        all_poses.add(start_pose_key)
        
        pose_indices = {}
        for pose_key in all_poses:
            pose_indices[pose_key] = self._parse_pose_key(pose_key)
        
        neighbors_cache = {}
        for pose_key, indices in pose_indices.items():
            neighbors = []
            for other_key in self.anchor_poses:
                if pose_key != other_key:
                    other_indices = pose_indices[other_key]
                    distance = self._index_distance(indices, other_indices)
                    if distance <= step_limit:
                        neighbors.append((other_key, distance))
            neighbors_cache[pose_key] = neighbors
        return neighbors_cache
    
    def _get_valid_actions(self, current_pose, remaining_steps):
        if current_pose not in self.neighbors_cache:
            return []
        valid_actions = []
        for neighbor_key, distance in self.neighbors_cache[current_pose]:
            if distance <= remaining_steps:
                valid_actions.append((neighbor_key, distance))
        return valid_actions
    
    def _calculate_anchor_reward(self, anchor_path):
        """只计算anchor poses的覆盖率"""
        if not anchor_path:
            return 0.0
        
        total_coverage = torch.zeros_like(next(iter(self.pose_coverage_cache.values())), device=self.device)
        for pose_key in anchor_path:
            if pose_key in self.pose_coverage_cache:
                total_coverage = total_coverage | self.pose_coverage_cache[pose_key]
        
        coverage_count = total_coverage.sum().item()
        coverage_ratio = coverage_count / self.max_anchor_coverage if self.max_anchor_coverage > 0 else 0.0
        
        # 更新当前最大coverage（用于UCB归一化）
        self.current_max_coverage = max(self.current_max_coverage, coverage_ratio)
        
        return coverage_ratio
    
    def _calculate_anchor_distances(self, anchor_path):
        """计算anchor path中相邻poses的距离列表"""
        if len(anchor_path) <= 1:
            return []
        
        distances = []
        for i in range(len(anchor_path) - 1):
            current_indices = self._parse_pose_key(anchor_path[i])
            next_indices = self._parse_pose_key(anchor_path[i + 1])
            distance = self._index_distance(current_indices, next_indices)
            distances.append(distance)
        
        return distances
    
    def _calculate_total_steps(self, anchor_path):
        """计算anchor path的总步数"""
        distances = self._calculate_anchor_distances(anchor_path)
        return sum(distances)
    
    def _simulate_random_path(self, node):
        current_pose = node.pose_key
        remaining_steps = node.remaining_steps
        anchor_path = node.path.copy()
        
        while remaining_steps > 0:
            valid_actions = self._get_valid_actions(current_pose, remaining_steps)
            if not valid_actions:
                break
            next_anchor, step_cost = random.choice(valid_actions)
            anchor_path.append(next_anchor)
            current_pose = next_anchor
            remaining_steps -= step_cost
        
        # 只计算anchor poses的覆盖率
        reward = self._calculate_anchor_reward(anchor_path)
        
        # 跟踪最优路径
        total_steps = self._calculate_total_steps(anchor_path)
        if not hasattr(self, 'best_anchor_path') or reward > self.best_anchor_reward:
            self.best_anchor_path = anchor_path
            self.best_anchor_reward = reward
            self.best_total_steps = total_steps
            self.best_distances = self._calculate_anchor_distances(anchor_path)
        
        return reward
    
    def _select(self, root):
        node = root
        while node.children and node.is_fully_expanded(self._get_valid_actions(node.pose_key, node.remaining_steps)):
            node = node.best_child(max_coverage=self.current_max_coverage)
        return node
    
    def _expand(self, node):
        valid_actions = self._get_valid_actions(node.pose_key, node.remaining_steps)
        if not valid_actions:
            return node
        
        unexplored_actions = []
        for next_pose, step_cost in valid_actions:
            if next_pose not in node.children:
                unexplored_actions.append((next_pose, step_cost))
        if not unexplored_actions:
            return node
        
        next_pose, step_cost = random.choice(unexplored_actions)
        new_path = node.path + [next_pose]
        new_remaining = node.remaining_steps - step_cost
        
        # 只考虑anchor poses的覆盖
        new_coverage = torch.zeros_like(next(iter(self.pose_coverage_cache.values())), device=self.device)
        for pose_key in new_path:
            if pose_key in self.pose_coverage_cache:
                new_coverage = new_coverage | self.pose_coverage_cache[pose_key]
        
        child_node = MCTSNode(next_pose, new_path, new_remaining, new_coverage, parent=node)
        node.add_child(next_pose, child_node)
        return child_node
    
    def _backpropagate(self, node, reward):
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent
    
    def plan_optimal_path(self, start_pose_key, max_steps, simulation_count=1000):
        print(f"Starting MCTS planning from {start_pose_key}, max_steps={max_steps}")
        
        self.neighbors_cache = self._build_neighbors_cache(start_pose_key)
        
        if start_pose_key not in self.neighbors_cache or not self.neighbors_cache[start_pose_key]:
            # 计算起始pose的覆盖率
            start_coverage = self._calculate_anchor_reward([start_pose_key])
            return {
                'anchor_path': [start_pose_key], 
                'anchor_distances': [], 
                'total_steps': 0, 
                'coverage': start_coverage,
                'coverage_points': int(start_coverage * self.max_anchor_coverage)
            }
        
        # 初始化
        if start_pose_key in self.pose_coverage_cache:
            initial_coverage = self.pose_coverage_cache[start_pose_key].clone()
        else:
            sample_mask = next(iter(self.pose_coverage_cache.values()))
            initial_coverage = torch.zeros_like(sample_mask, device=self.device)
        
        root = MCTSNode(start_pose_key, [start_pose_key], max_steps, initial_coverage)
        
        # 初始化最优路径跟踪
        self.best_anchor_path = [start_pose_key]
        self.best_anchor_reward = self._calculate_anchor_reward([start_pose_key])
        self.best_total_steps = 0
        self.best_distances = []
        self.current_max_coverage = self.best_anchor_reward
        
        # MCTS主循环
        for i in range(simulation_count):
            if (i + 1) % 1000 == 0:
                current_best_coverage_points = int(self.best_anchor_reward * self.max_anchor_coverage)
                print(f"MCTS simulation {i + 1}/{simulation_count}, best coverage: {current_best_coverage_points}/{self.max_anchor_coverage} ({self.best_anchor_reward*100:.1f}%)")               
            
            leaf_node = self._select(root)
            new_node = self._expand(leaf_node)
            reward = self._simulate_random_path(new_node)
            self._backpropagate(new_node, reward)
        
        final_coverage_points = int(self.best_anchor_reward * self.max_anchor_coverage)
        
        print(f"MCTS completed:")
        print(f"  Anchor path: {len(self.best_anchor_path)} poses")
        print(f"  Total steps: {self.best_total_steps}")
        print(f"  Anchor distances: {self.best_distances}")
        print(f"  Coverage: {final_coverage_points}/{self.max_anchor_coverage} ({self.best_anchor_reward*100:.1f}%)")
        
        return {
            'anchor_path': self.best_anchor_path,
            'anchor_distances': self.best_distances,
            'total_steps': self.best_total_steps,
            'coverage': self.best_anchor_reward,
            'coverage_points': final_coverage_points
        }

def plan_path_with_mcts(anchor_poses, pose_coverage_cache, start_pose, max_steps, simulation_count=1000, device='cuda'):
    """
    使用MCTS规划最优的anchor pose路径
    
    Args:
        anchor_poses: 可选择的anchor poses列表
        pose_coverage_cache: pose覆盖信息缓存
        start_pose: 起始pose
        max_steps: 最大步数限制
        simulation_count: MCTS仿真次数
        device: 计算设备
    
    Returns:
        dict: {
            'anchor_path': 最优anchor pose路径,
            'anchor_distances': 相邻anchor poses的距离列表,
            'total_steps': 总步数,
            'coverage': 覆盖率(0-1),
            'coverage_points': 覆盖的点数
        }
    """
    planner = MCTSPathPlanner(anchor_poses, pose_coverage_cache, device)
    result = planner.plan_optimal_path(start_pose, max_steps, simulation_count)
    return result

'''

########################################################################nbv
'''
import torch
import numpy as np
import random
import math
from collections import defaultdict

class MCTSNode:
    def __init__(self, pose_key, path, remaining_steps, coverage_mask, parent=None):
        self.pose_key = pose_key
        self.path = path  # anchor poses路径
        self.remaining_steps = remaining_steps
        self.coverage_mask = coverage_mask
        self.parent = parent
        
        self.visit_count = 0
        self.total_reward = 0.0
        self.children = {}
        
    def is_fully_expanded(self, valid_actions):
        return len(self.children) == len(valid_actions)
    
    def best_child(self, c=1.4, max_coverage=None):
        choices = []
        for action, child in self.children.items():
            if child.visit_count == 0:
                ucb_value = float('inf')
            else:
                # 归一化q_value
                q_value = child.total_reward / child.visit_count
                if max_coverage is not None and max_coverage > 0:
                    q_value = q_value / max_coverage
                
                ucb_value = q_value + c * math.sqrt(2 * math.log(self.visit_count) / child.visit_count)
            choices.append((ucb_value, action, child))
        return max(choices, key=lambda x: x[0])[2]
    
    def add_child(self, action, child_node):
        self.children[action] = child_node

class MCTSPathPlanner:
    def __init__(self, anchor_poses, pose_coverage_cache, device='cuda'):
        self.anchor_poses = set(anchor_poses)
        self.pose_coverage_cache = pose_coverage_cache
        self.device = device
        self.neighbors_cache = {}
        
        # 计算anchor poses的总覆盖（不包括中间poses）
        anchor_coverage = torch.zeros_like(next(iter(pose_coverage_cache.values())), device=device)
        for pose_key in anchor_poses:
            if pose_key in pose_coverage_cache:
                anchor_coverage = anchor_coverage | pose_coverage_cache[pose_key]
        self.max_anchor_coverage = anchor_coverage.sum().item()
        
        # 跟踪当前simulation阶段的最大coverage（用于UCB归一化）
        self.current_max_coverage = 0.0
        
        print(f"Maximum possible anchor coverage: {self.max_anchor_coverage} points")
    
    def _parse_pose_key(self, pose_key):
        import ast
        return ast.literal_eval(pose_key)
    
    def _index_distance(self, indices1, indices2):
        """计算两个pose索引之间的步数距离（部分维度考虑环形拓扑）"""
        max_values = [8, 4, 6, 5, 10]
        is_circular = [False, False, False, True, True]  # 只有后两个维度是环形的
        
        total_distance = 0
        for i, (a, b) in enumerate(zip(indices1, indices2)):
            if i < len(max_values):
                max_val = max_values[i]
                if is_circular[i]:
                    # 环形距离：考虑wrap-around
                    direct_dist = abs(a - b)
                    wrap_dist = max_val - direct_dist
                    min_dist = min(direct_dist, wrap_dist)
                    total_distance += min_dist
                else:
                    # 普通线性距离
                    total_distance += abs(a - b)
            else:
                # 如果有额外维度，使用普通距离
                total_distance += abs(a - b)
        
        return total_distance
    
    def _build_neighbors_cache(self, start_pose_key, step_limit=3):
        all_poses = set(self.anchor_poses)
        all_poses.add(start_pose_key)
        
        pose_indices = {}
        for pose_key in all_poses:
            pose_indices[pose_key] = self._parse_pose_key(pose_key)
        
        neighbors_cache = {}
        for pose_key, indices in pose_indices.items():
            neighbors = []
            for other_key in self.anchor_poses:
                if pose_key != other_key:
                    other_indices = pose_indices[other_key]
                    distance = self._index_distance(indices, other_indices)
                    if distance <= step_limit:
                        neighbors.append((other_key, distance))
            neighbors_cache[pose_key] = neighbors
        return neighbors_cache
    
    def _get_valid_actions(self, current_pose, remaining_steps):
        if current_pose not in self.neighbors_cache:
            return []
        valid_actions = []
        for neighbor_key, distance in self.neighbors_cache[current_pose]:
            if distance <= remaining_steps:
                valid_actions.append((neighbor_key, distance))
        return valid_actions
    
    def _calculate_anchor_reward(self, anchor_path):
        """只计算anchor poses的覆盖率"""
        if not anchor_path:
            return 0.0
        
        total_coverage = torch.zeros_like(next(iter(self.pose_coverage_cache.values())), device=self.device)
        for pose_key in anchor_path:
            if pose_key in self.pose_coverage_cache:
                total_coverage = total_coverage | self.pose_coverage_cache[pose_key]
        
        coverage_count = total_coverage.sum().item()
        coverage_ratio = coverage_count / self.max_anchor_coverage if self.max_anchor_coverage > 0 else 0.0
        
        # 更新当前最大coverage（用于UCB归一化）
        self.current_max_coverage = max(self.current_max_coverage, coverage_ratio)
        
        return coverage_ratio
    
    def _calculate_anchor_distances(self, anchor_path):
        """计算anchor path中相邻poses的距离列表"""
        if len(anchor_path) <= 1:
            return []
        
        distances = []
        for i in range(len(anchor_path) - 1):
            current_indices = self._parse_pose_key(anchor_path[i])
            next_indices = self._parse_pose_key(anchor_path[i + 1])
            distance = self._index_distance(current_indices, next_indices)
            distances.append(distance)
        
        return distances
    
    def _calculate_total_steps(self, anchor_path):
        """计算anchor path的总步数"""
        distances = self._calculate_anchor_distances(anchor_path)
        return sum(distances)
    
    def _simulate_random_path(self, node):
        current_pose = node.pose_key
        remaining_steps = node.remaining_steps
        anchor_path = node.path.copy()
        
        while remaining_steps > 0:
            valid_actions = self._get_valid_actions(current_pose, remaining_steps)
            if not valid_actions:
                break
            next_anchor, step_cost = random.choice(valid_actions)
            anchor_path.append(next_anchor)
            current_pose = next_anchor
            remaining_steps -= step_cost
        
        # 只计算anchor poses的覆盖率
        reward = self._calculate_anchor_reward(anchor_path)
        
        # 跟踪最优路径
        total_steps = self._calculate_total_steps(anchor_path)
        if not hasattr(self, 'best_anchor_path') or reward > self.best_anchor_reward:
            self.best_anchor_path = anchor_path
            self.best_anchor_reward = reward
            self.best_total_steps = total_steps
            self.best_distances = self._calculate_anchor_distances(anchor_path)
        
        return reward
    
    def _select(self, root):
        node = root
        while node.children and node.is_fully_expanded(self._get_valid_actions(node.pose_key, node.remaining_steps)):
            node = node.best_child(max_coverage=self.current_max_coverage)
        return node
    
    def _expand(self, node):
        valid_actions = self._get_valid_actions(node.pose_key, node.remaining_steps)
        if not valid_actions:
            return node
        
        unexplored_actions = []
        for next_pose, step_cost in valid_actions:
            if next_pose not in node.children:
                unexplored_actions.append((next_pose, step_cost))
        if not unexplored_actions:
            return node
        
        next_pose, step_cost = random.choice(unexplored_actions)
        new_path = node.path + [next_pose]
        new_remaining = node.remaining_steps - step_cost
        
        # 只考虑anchor poses的覆盖
        new_coverage = torch.zeros_like(next(iter(self.pose_coverage_cache.values())), device=self.device)
        for pose_key in new_path:
            if pose_key in self.pose_coverage_cache:
                new_coverage = new_coverage | self.pose_coverage_cache[pose_key]
        
        child_node = MCTSNode(next_pose, new_path, new_remaining, new_coverage, parent=node)
        node.add_child(next_pose, child_node)
        return child_node
    
    def _backpropagate(self, node, reward):
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent
    
    def _run_nbv_simulation(self, start_pose_key, max_steps):
        """NBV贪心算法：每步选择最大coverage gain的pose"""
        print("Running NBV simulation...")
        
        current_pose = start_pose_key
        path = [start_pose_key]
        remaining_steps = max_steps
        
        # 累积覆盖
        cumulative_coverage = torch.zeros_like(next(iter(self.pose_coverage_cache.values())), device=self.device)
        if start_pose_key in self.pose_coverage_cache:
            cumulative_coverage = cumulative_coverage | self.pose_coverage_cache[start_pose_key]
        
        while remaining_steps > 0:
            valid_actions = self._get_valid_actions(current_pose, remaining_steps)
            if not valid_actions:
                break
                
            best_gain = 0
            best_pose = None
            best_cost = 0
            
            # 找到coverage gain最大的pose
            for next_pose, step_cost in valid_actions:
                if next_pose in self.pose_coverage_cache:
                    new_coverage = self.pose_coverage_cache[next_pose]
                    coverage_gain = (new_coverage & ~cumulative_coverage).sum().item()
                    
                    if coverage_gain > best_gain:
                        best_gain = coverage_gain
                        best_pose = next_pose
                        best_cost = step_cost
            
            if best_pose is None:
                break
                
            path.append(best_pose)
            cumulative_coverage = cumulative_coverage | self.pose_coverage_cache[best_pose]
            current_pose = best_pose
            remaining_steps -= best_cost
        
        # 计算NBV结果
        nbv_reward = self._calculate_anchor_reward(path)
        nbv_distances = self._calculate_anchor_distances(path)
        nbv_steps = sum(nbv_distances)
        
        print(f"NBV result: {len(path)} poses, {nbv_steps} steps, {nbv_reward*100:.1f}% coverage")
        
        return {
            'anchor_path': path,
            'anchor_distances': nbv_distances,
            'total_steps': nbv_steps,
            'coverage': nbv_reward,
            'coverage_points': int(nbv_reward * self.max_anchor_coverage)
        }

    def plan_optimal_path(self, start_pose_key, max_steps, simulation_count=1000, use_nbv_initialization=False):
        print(f"Starting planning from {start_pose_key}, max_steps={max_steps}")
        
        self.neighbors_cache = self._build_neighbors_cache(start_pose_key)
        
        if start_pose_key not in self.neighbors_cache or not self.neighbors_cache[start_pose_key]:
            start_coverage = self._calculate_anchor_reward([start_pose_key])
            result = {
                'anchor_path': [start_pose_key], 
                'anchor_distances': [], 
                'total_steps': 0, 
                'coverage': start_coverage,
                'coverage_points': int(start_coverage * self.max_anchor_coverage)
            }
            return {'nbv_result': result, 'mcts_result': result} if use_nbv_initialization else result
        
        # NBV初始化（可选）
        nbv_result = None
        if use_nbv_initialization:
            nbv_result = self._run_nbv_simulation(start_pose_key, max_steps)
        
        # MCTS初始化
        if start_pose_key in self.pose_coverage_cache:
            initial_coverage = self.pose_coverage_cache[start_pose_key].clone()
        else:
            sample_mask = next(iter(self.pose_coverage_cache.values()))
            initial_coverage = torch.zeros_like(sample_mask, device=self.device)
        
        root = MCTSNode(start_pose_key, [start_pose_key], max_steps, initial_coverage)
        
        # 初始化最优路径跟踪
        if use_nbv_initialization and nbv_result:
            self.best_anchor_path = nbv_result['anchor_path']
            self.best_anchor_reward = nbv_result['coverage']
            self.best_total_steps = nbv_result['total_steps']
            self.best_distances = nbv_result['anchor_distances']
            self.current_max_coverage = nbv_result['coverage']
            print(f"MCTS initialized with NBV result: {self.best_anchor_reward*100:.1f}% coverage")
        else:
            self.best_anchor_path = [start_pose_key]
            self.best_anchor_reward = self._calculate_anchor_reward([start_pose_key])
            self.best_total_steps = 0
            self.best_distances = []
            self.current_max_coverage = self.best_anchor_reward
        
        # MCTS主循环
        print("Starting MCTS simulation...")
        for i in range(simulation_count):
            if (i + 1) % 1000 == 0:
                current_best_coverage_points = int(self.best_anchor_reward * self.max_anchor_coverage)
                print(f"MCTS simulation {i + 1}/{simulation_count}, best coverage: {current_best_coverage_points}/{self.max_anchor_coverage} ({self.best_anchor_reward*100:.1f}%)")               
            
            leaf_node = self._select(root)
            new_node = self._expand(leaf_node)
            reward = self._simulate_random_path(new_node)
            self._backpropagate(new_node, reward)
        
        final_coverage_points = int(self.best_anchor_reward * self.max_anchor_coverage)
        
        print(f"MCTS completed:")
        print(f"  Anchor path: {len(self.best_anchor_path)} poses")
        print(f"  Total steps: {self.best_total_steps}")
        print(f"  Anchor distances: {self.best_distances}")
        print(f"  Coverage: {final_coverage_points}/{self.max_anchor_coverage} ({self.best_anchor_reward*100:.1f}%)")
        
        mcts_result = {
            'anchor_path': self.best_anchor_path,
            'anchor_distances': self.best_distances,
            'total_steps': self.best_total_steps,
            'coverage': self.best_anchor_reward,
            'coverage_points': final_coverage_points
        }
        
        if use_nbv_initialization:
            return {'nbv_result': nbv_result, 'mcts_result': mcts_result}
        else:
            return mcts_result

def plan_path_with_mcts(anchor_poses, pose_coverage_cache, start_pose, max_steps, simulation_count=1000, use_nbv_initialization=False, device='cuda'):
    """
    使用MCTS规划最优的anchor pose路径
    
    Args:
        anchor_poses: 可选择的anchor poses列表
        pose_coverage_cache: pose覆盖信息缓存
        start_pose: 起始pose
        max_steps: 最大步数限制
        simulation_count: MCTS仿真次数
        use_nbv_initialization: 是否使用NBV初始化
        device: 计算设备
    
    Returns:
        dict: 如果use_nbv_initialization=False:
            {
                'anchor_path': 最优anchor pose路径,
                'anchor_distances': 相邻anchor poses的距离列表,
                'total_steps': 总步数,
                'coverage': 覆盖率(0-1),
                'coverage_points': 覆盖的点数
            }
        如果use_nbv_initialization=True:
            {
                'nbv_result': NBV贪心结果,
                'mcts_result': MCTS+NBV优化结果
            }
    """
    planner = MCTSPathPlanner(anchor_poses, pose_coverage_cache, device)
    result = planner.plan_optimal_path(start_pose, max_steps, simulation_count, use_nbv_initialization)
    return result

'''
import torch
import numpy as np
import random
import math
from collections import defaultdict

class MCTSNode:
    def __init__(self, pose_key, path, remaining_steps, parent=None):
        self.pose_key = pose_key
        self.path = path  # anchor poses路径
        self.remaining_steps = remaining_steps
        self.parent = parent
        
        self.visit_count = 0
        self.total_reward = 0.0
        self.children = {}
        
    def is_fully_expanded(self, valid_actions):
        return len(self.children) == len(valid_actions)
    
    def best_child(self, c=1.4*10):
        choices = []
        for action, child in self.children.items():
            if child.visit_count == 0:
                ucb_value = float('inf')
            else:
                q_value = child.total_reward / child.visit_count
                ucb_value = q_value + c * math.sqrt(2 * math.log(self.visit_count) / child.visit_count)
            choices.append((ucb_value, action, child))
        return max(choices, key=lambda x: x[0])[2]
    
    def add_child(self, action, child_node):
        self.children[action] = child_node

class MCTSPathPlanner:
    def __init__(self, anchor_poses, pose_coverage_cache, device='cuda'):
        self.anchor_poses = set(anchor_poses)
        self.pose_coverage_cache = pose_coverage_cache
        self.device = device
        self.neighbors_cache = {}
        
        # 计算anchor poses的总覆盖
        anchor_coverage = torch.zeros_like(next(iter(pose_coverage_cache.values())), device=device)
        for pose_key in anchor_poses:
            if pose_key in pose_coverage_cache:
                anchor_coverage = anchor_coverage | pose_coverage_cache[pose_key]
        self.max_anchor_coverage = anchor_coverage.sum().item()
        
        print(f"Maximum possible anchor coverage: {self.max_anchor_coverage} points")
    
    def _parse_pose_key(self, pose_key):
        import ast
        return ast.literal_eval(pose_key)
    
    def _index_distance(self, indices1, indices2):
        """计算两个pose索引之间的步数距离（部分维度考虑环形拓扑）"""
        max_values = [8, 4, 6, 5, 10]
        is_circular = [False, False, False, True, True]  # 只有后两个维度是环形的
        
        total_distance = 0
        for i, (a, b) in enumerate(zip(indices1, indices2)):
            if i < len(max_values):
                max_val = max_values[i]
                if is_circular[i]:
                    # 环形距离：考虑wrap-around
                    direct_dist = abs(a - b)
                    wrap_dist = max_val - direct_dist
                    min_dist = min(direct_dist, wrap_dist)
                    total_distance += min_dist
                else:
                    # 普通线性距离
                    total_distance += abs(a - b)
            else:
                # 如果有额外维度，使用普通距离
                total_distance += abs(a - b)
        
        return total_distance
    
    def _build_neighbors_cache(self, start_pose_key, step_limit=1):
        all_poses = set(self.anchor_poses)
        all_poses.add(start_pose_key)
        
        pose_indices = {}
        for pose_key in all_poses:
            pose_indices[pose_key] = self._parse_pose_key(pose_key)
        
        neighbors_cache = {}
        for pose_key, indices in pose_indices.items():
            neighbors = []
            for other_key in self.anchor_poses:
                if pose_key != other_key:
                    other_indices = pose_indices[other_key]
                    distance = self._index_distance(indices, other_indices)
                    if distance == step_limit:
                        neighbors.append((other_key, distance))
            neighbors_cache[pose_key] = neighbors
        return neighbors_cache
    
    def _get_valid_actions(self, current_pose, remaining_steps):
        if current_pose not in self.neighbors_cache:
            return []
        valid_actions = []
        for neighbor_key, distance in self.neighbors_cache[current_pose]:
            if distance <= remaining_steps:
                valid_actions.append((neighbor_key, distance))
        return valid_actions
    
    def _calculate_path_coverage(self, anchor_path):
        """计算路径的覆盖率"""
        if not anchor_path:
            return 0.0
        
        total_coverage = torch.zeros_like(next(iter(self.pose_coverage_cache.values())), device=self.device)
        for pose_key in anchor_path:
            if pose_key in self.pose_coverage_cache:
                total_coverage = total_coverage | self.pose_coverage_cache[pose_key]
        
        coverage_count = total_coverage.sum().item()
        coverage_ratio = coverage_count / self.max_anchor_coverage if self.max_anchor_coverage > 0 else 0.0
        return coverage_ratio
    
    def _calculate_path_distances(self, anchor_path):
        """计算路径中相邻poses的距离列表"""
        if len(anchor_path) <= 1:
            return []
        
        distances = []
        for i in range(len(anchor_path) - 1):
            current_indices = self._parse_pose_key(anchor_path[i])
            next_indices = self._parse_pose_key(anchor_path[i + 1])
            distance = self._index_distance(current_indices, next_indices)
            distances.append(distance)
        
        return distances
    
    def _run_nbv_from_state(self, start_pose_key, remaining_steps, existing_path):
        """从给定状态运行NBV算法"""
        # 计算已有路径的覆盖
        cumulative_coverage = torch.zeros_like(next(iter(self.pose_coverage_cache.values())), device=self.device)
        for pose_key in existing_path:
            if pose_key in self.pose_coverage_cache:
                cumulative_coverage = cumulative_coverage | self.pose_coverage_cache[pose_key]
        
        current_pose = start_pose_key
        nbv_extension = []
        steps_left = remaining_steps
        
        while steps_left > 0:
            valid_actions = self._get_valid_actions(current_pose, steps_left)
            if not valid_actions:
                break
                
            best_gain = -1
            best_pose = None
            best_cost = 0
            
            # 找到最大coverage gain的pose
            for next_pose, step_cost in valid_actions:
                if next_pose in self.pose_coverage_cache:
                    new_coverage = self.pose_coverage_cache[next_pose]
                    coverage_gain = (new_coverage & ~cumulative_coverage).sum().item()
                    
                    if coverage_gain > best_gain:
                        best_gain = coverage_gain
                        best_pose = next_pose
                        best_cost = step_cost
            
            if best_pose is None:
                break
                
            nbv_extension.append(best_pose)
            cumulative_coverage = cumulative_coverage | self.pose_coverage_cache[best_pose]
            current_pose = best_pose
            steps_left -= best_cost
        
        # 返回完整路径和覆盖率
        full_path = existing_path + nbv_extension
        coverage_ratio = self._calculate_path_coverage(full_path)
        
        return coverage_ratio, full_path
    
    def _select(self, root):
        node = root
        while node.children and node.is_fully_expanded(self._get_valid_actions(node.pose_key, node.remaining_steps)):
            node = node.best_child()
        return node
    
    def _expand(self, node):
        valid_actions = self._get_valid_actions(node.pose_key, node.remaining_steps)
        if not valid_actions:
            return node
        
        unexplored_actions = []
        for next_pose, step_cost in valid_actions:
            if next_pose not in node.children:
                unexplored_actions.append((next_pose, step_cost))
        if not unexplored_actions:
            return node
        
        # 随机选择一个未探索的动作
        next_pose, step_cost = random.choice(unexplored_actions)
        
        # 创建子节点
        new_path = node.path + [next_pose]
        new_remaining = node.remaining_steps - step_cost
        child_node = MCTSNode(next_pose, new_path, new_remaining, parent=node)
        
        # 从子节点运行NBV评估
        nbv_coverage, nbv_full_path = self._run_nbv_from_state(next_pose, new_remaining, new_path)
        
        # 更新全局最佳路径
        if nbv_coverage > self.best_coverage:
            self.best_path = nbv_full_path
            self.best_coverage = nbv_coverage
            self.best_distances = self._calculate_path_distances(nbv_full_path)
            self.best_steps = sum(self.best_distances)
        
        # Backpropagate NBV结果
        self._backpropagate(child_node, nbv_coverage)
        node.add_child(next_pose, child_node)
        return child_node
    
    def _backpropagate(self, node, reward):
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent
    
    def plan_optimal_path(self, start_pose_key, max_steps, simulation_count=1000):
        print(f"Starting planning from {start_pose_key}, max_steps={max_steps}")
        
        self.neighbors_cache = self._build_neighbors_cache(start_pose_key)
        
        if start_pose_key not in self.neighbors_cache or not self.neighbors_cache[start_pose_key]:
            start_coverage = self._calculate_path_coverage([start_pose_key])
            result = {
                'anchor_path': [start_pose_key], 
                'anchor_distances': [], 
                'total_steps': 0, 
                'coverage': start_coverage,
                'coverage_points': int(start_coverage * self.max_anchor_coverage)
            }
            return result
        
        # NBV初始化 (必须)
        print("Running NBV initialization...")
        nbv_coverage, nbv_path = self._run_nbv_from_state(start_pose_key, max_steps, [start_pose_key])
        
        # 初始化全局最佳路径
        self.best_path = nbv_path
        self.best_coverage = nbv_coverage
        self.best_distances = self._calculate_path_distances(nbv_path)
        self.best_steps = sum(self.best_distances)
        
        print(f"NBV initialization: {len(nbv_path)} poses, {self.best_steps} steps, {nbv_coverage*100:.1f}% coverage")
        
        # MCTS搜索
        root = MCTSNode(start_pose_key, [start_pose_key], max_steps)
        
        print("Starting MCTS search...")
        for i in range(simulation_count):
            if (i + 1) % 1000 == 0:
                current_best_coverage_points = int(self.best_coverage * self.max_anchor_coverage)
                print(f"MCTS iteration {i + 1}/{simulation_count}, best coverage: {current_best_coverage_points}/{self.max_anchor_coverage} ({self.best_coverage*100:.1f}%)")
            
            # MCTS步骤：Selection -> Expansion (包含NBV评估) -> Backpropagation
            leaf_node = self._select(root)
            self._expand(leaf_node)
        
        final_coverage_points = int(self.best_coverage * self.max_anchor_coverage)
        
        print(f"MCTS completed:")
        print(f"  Final path: {len(self.best_path)} poses")
        print(f"  Total steps: {self.best_steps}")
        print(f"  Coverage: {final_coverage_points}/{self.max_anchor_coverage} ({self.best_coverage*100:.1f}%)")
        
        return {
            'anchor_path': self.best_path,
            'anchor_distances': self.best_distances,
            'total_steps': self.best_steps,
            'coverage': self.best_coverage,
            'coverage_points': final_coverage_points
        }

def plan_path_with_mcts(anchor_poses, pose_coverage_cache, start_pose, max_steps, simulation_count=1000, device='cuda'):
    """
    使用优化后的MCTS规划最优的anchor pose路径
    
    Args:
        anchor_poses: 可选择的anchor poses列表
        pose_coverage_cache: pose覆盖信息缓存
        start_pose: 起始pose
        max_steps: 最大步数限制
        simulation_count: MCTS仿真次数
        device: 计算设备
    
    Returns:
        dict: {
            'anchor_path': 最优anchor pose路径,
            'anchor_distances': 相邻anchor poses的距离列表,
            'total_steps': 总步数,
            'coverage': 覆盖率(0-1),
            'coverage_points': 覆盖的点数
        }
    """
    planner = MCTSPathPlanner(anchor_poses, pose_coverage_cache, device)
    result = planner.plan_optimal_path(start_pose, max_steps, simulation_count)
    return result