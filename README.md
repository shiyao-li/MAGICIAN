<div align="center">
<h1>MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping</h1>

[Shiyao Li](https://shiyao-li.github.io/), [Antoine Guédon](https://anttwo.github.io/), [Shizhe Chen](https://cshizhe.github.io/), [Vincent Lepetit](https://vincentlepetit.github.io/)

<a href="https://arxiv.org/abs/2603.22650" style="margin-right: 10px;">
  <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white" alt="arXiv Paper">
</a>
<a href="https://shiyao-li.github.io/magician/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

<br>
<img src="assets/image.png" alt="Teaser" width="100%">

</div>

## Getting Started

### 1. Clone the Repository

```bash
git clone --recursive git@github.com:shiyao-li/MAGICIAN.git
cd MAGICIAN
```

### 2. Environment Setup

```bash
conda env create -f environment.yml
conda activate magician
cd RaDe-GS
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/warp-patch-ncc --no-build-isolation
pip install submodules/simple-knn/ --no-build-isolation
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation

# tetra-nerf for Marching Tetrahedra
conda install conda-forge::cgal
pip install submodules/tetra_triangulation/ --no-build-isolation
```

### 3. Dataset

Download the dataset [here](https://huggingface.co/datasets/sli016/scenes/resolve/main/macarons%2B%2B.zip) and place it under a `data/` folder in the project root.

### 4. Pretrained Weights

Download the pretrained model weights from [Google Drive](https://drive.google.com/drive/folders/1wyc9_QFmcxOz4oerE8kCQ3I8LO5zioZL) and place them under a `weights/` folder in the project root.

### 5. Run

```bash
python test_magician_planning.py
```

### 6. Configuration

Key parameters are in `configs/test/test_in_default_scenes_config.json`:

| Parameter | Description |
|-----------|-------------|
| `beam_width` | Beam search: number of candidates kept at each step |
| `beam_steps` | Beam search: lookahead depth (number of steps) |
| `lmdb_dir_name` | Name of the output LMDB directory under `results/scene_exploration/` |

### 7. Evaluate Metrics

```bash
python evaluation_lmdb.py
```

The LMDB file (specified by `lmdb_dir_name`) stores the following data for each trajectory:

- **Coverage update history**: how coverage evolves step by step
- **Camera poses**: the full history of visited camera positions
- **Final point cloud**: the reconstructed point cloud at the end of the trajectory

## Citation

```bibtex
@inproceedings{li2026magician,
  author = "Shiyao Li and Antoine Guédon and Shizhe Chen and Vincent Lepetit",
  title = {{MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping}},
  booktitle = {{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}},
  year = 2026
}
```