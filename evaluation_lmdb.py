"""
Compute metrics from LMDB database.
"""

import lmdb
import pickle
import numpy as np
from collections import defaultdict
import argparse


def load_from_lmdb(lmdb_env, key):
    """Load data from LMDB database."""
    with lmdb_env.begin() as txn:
        serialized_data = txn.get(key.encode('utf-8'))
        if serialized_data is None:
            return None
        return pickle.loads(serialized_data)


def compute_auc(coverage_list):
    """Compute Area Under the Curve using trapezoidal rule."""
    x = np.arange(len(coverage_list))
    y = np.array(coverage_list)
    return np.trapz(y, x)


def analyze_lmdb_coverage(lmdb_dir):
    """Analyze coverage data from LMDB database."""
    # Open LMDB
    lmdb_env = lmdb.open(lmdb_dir, readonly=True, lock=False)

    # Collect all keys
    all_keys = []
    with lmdb_env.begin() as txn:
        cursor = txn.cursor()
        for key, _ in cursor:
            key_str = key.decode('utf-8')
            if '/point_cloud' not in key_str:
                all_keys.append(key_str)

    # Group by scene
    scene_data = defaultdict(dict)
    for key in all_keys:
        parts = key.split('/')
        if len(parts) != 2:
            continue
        scene_name = parts[0]
        traj_idx = int(parts[1])

        data = load_from_lmdb(lmdb_env, key)
        if data and 'coverage' in data:
            scene_data[scene_name][traj_idx] = data['coverage']

    lmdb_env.close()

    # Collect all AUC and max coverage values across all scenes
    all_auc_values = []
    all_max_cov_values = []

    # Analyze each scene
    for scene_name in sorted(scene_data.keys()):
        trajectories = scene_data[scene_name]
        print(f"\n{scene_name}: {len(trajectories)} trajectories")

        auc_values = []
        max_cov_values = []

        for traj_idx in sorted(trajectories.keys()):
            coverage = trajectories[traj_idx]

            # Start from index 1
            coverage_from_idx1 = coverage[:100]
            # Compute AUC and max coverage
            auc = compute_auc(coverage_from_idx1)
            max_cov = max(coverage_from_idx1)

            auc_values.append(auc)
            max_cov_values.append(max_cov)

            print(f"  Traj {traj_idx}: AUC={auc:.2f}, Max Cov={max_cov:.4f}")

        # Scene averages
        avg_auc = np.mean(auc_values)
        avg_max_cov = np.mean(max_cov_values)
        print(f"  Average: AUC={avg_auc:.2f}, Max Cov={avg_max_cov:.4f}")

        # Accumulate for overall statistics
        all_auc_values.extend(auc_values)
        all_max_cov_values.extend(max_cov_values)

    # Overall statistics across all scenes
    if all_auc_values:
        overall_avg_auc = np.mean(all_auc_values)
        overall_avg_max_cov = np.mean(all_max_cov_values)
        print(f"\n{'='*60}")
        print(f"Overall Statistics (all scenes):")
        print(f"  Average AUC: {overall_avg_auc:.2f}")
        print(f"  Average Max Cov: {overall_avg_max_cov:.4f}")
        print(f"  Total trajectories: {len(all_auc_values)}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Analyze LMDB coverage data')
    parser.add_argument('--lmdb_dir', type=str,
                        default='results/scene_exploration/magician_lmdb',
                        help='Path to LMDB directory')
    args = parser.parse_args()

    analyze_lmdb_coverage(args.lmdb_dir)


if __name__ == "__main__":
    main()
