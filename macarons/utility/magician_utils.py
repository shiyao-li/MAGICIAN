import os
import torch
import pickle
import lmdb
import shutil
import numpy as np
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

def line_segment_mesh_intersection(start_point, end_point, mesh):
    direction = end_point - start_point
    line_length = np.linalg.norm(direction)
    
    if line_length < 1e-6: 
        return False
    
    direction = direction / line_length
    
    locations, _, _ = mesh.intersects_location(
        ray_origins=[start_point],
        ray_directions=[direction]
    )
    
    if len(locations) == 0:
        return False
    
    distances = np.linalg.norm(locations - start_point, axis=1)
    return np.any(distances < line_length)

def cleanup_trajectory_folders(training_frames_path, keep_folders=['imgs']):
    """
    Delete all subfolders except the ones specified in keep_folders.
    Only keeps the imgs folder and deletes frames, depths, occupancy folders.

    Args:
        training_frames_path: Path to the frames folder (e.g., .../training/0/frames/)
        keep_folders: List of folder names to keep (default: ['imgs'])
    """
    # Get parent directory (e.g., .../training/0/)
    trajectory_dir = os.path.dirname(training_frames_path)

    if not os.path.exists(trajectory_dir):
        print(f"Warning: Trajectory directory does not exist: {trajectory_dir}")
        return

    deleted_folders = []
    kept_folders = []

    print(f"\n=== Cleaning up trajectory directory: {trajectory_dir} ===")

    # Iterate through all items in the trajectory directory
    for item in os.listdir(trajectory_dir):
        item_path = os.path.join(trajectory_dir, item)

        # Only process directories
        if os.path.isdir(item_path):
            if item in keep_folders:
                kept_folders.append(item)
                print(f"  ✓ Keeping folder: {item}")
            else:
                # Delete the folder and all its contents
                shutil.rmtree(item_path)
                deleted_folders.append(item)
                print(f"  ✗ Deleted folder: {item}")

    print(f"\nCleanup summary:")
    print(f"  - Kept: {kept_folders}")
    print(f"  - Deleted: {deleted_folders}\n")

def save_to_lmdb(lmdb_env, key, data_dict):
    """
    Save data to LMDB database.

    Args:
        lmdb_env: LMDB environment
        key: string key (e.g., "scene_name/start_cam_idx")
        data_dict: dictionary containing the data to save
    """
    with lmdb_env.begin(write=True) as txn:
        # Serialize data using pickle
        serialized_data = pickle.dumps(data_dict)
        txn.put(key.encode('utf-8'), serialized_data)
    print(f"Saved data to LMDB with key: {key}")

def load_from_lmdb(lmdb_env, key):
    """
    Load data from LMDB database.

    Args:
        lmdb_env: LMDB environment
        key: string key (e.g., "scene_name/start_cam_idx")

    Returns:
        data_dict: deserialized dictionary
    """
    with lmdb_env.begin() as txn:
        serialized_data = txn.get(key.encode('utf-8'))
        if serialized_data is None:
            return None
        return pickle.loads(serialized_data)


def clear_folder(folder_path):
    """
    remove all for a folder
    """
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print(f" {folder_path} is clear")