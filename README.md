
## Getting Started

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Dataset, Environment & Model Weights

Please follow the instructions from [MACARONS](https://github.com/Anttwo/MACARONS/) to set up:

- **Dataset** — download [here](https://huggingface.co/datasets/sli016/scenes/resolve/main/macarons%2B%2B.zip)
- **Conda environment** — create and activate the conda environment
- **Model weights** — download the pretrained model weights

### 3. Install RaDe-GS Dependencies

```bash
cd RaDe-GS
```

Then follow the installation instructions from [RaDe-GS](https://github.com/HKUST-SAIL/RaDe-GS) to install all required dependencies.

### 4. Run

```bash
python test_pai_planning.py
```