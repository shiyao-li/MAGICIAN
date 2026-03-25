<div align="center">
<h1>MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping</h1>

[Shiyao Li](https://shiyao-li.github.io/), [Antoine Guédon](https://anttwo.github.io/), [Shizhe Chen](https://cshizhe.github.io/), [Vincent Lepetit](https://vincentlepetit.github.io/)

<a href="https://arxiv.org/abs/2603.22650" style="margin-right: 10px;">
  <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white" alt="arXiv Paper">
</a>
<a href="https://shiyao-li.github.io/magician/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

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