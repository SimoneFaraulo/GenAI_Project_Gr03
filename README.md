# Generative AI Project - Conditional Image Generation

## Project Overview

This project implements and compares different Generative AI architectures for conditional image synthesis. The goal is to generate realistic human faces based on the **CelebA** dataset, enabling control over specific semantic attributes (gender, expression, age).

The system is designed with modularity in mind to support the training and inference of three distinct classes of generative models:

1. **Conditional Variational Autoencoder (C-VAE)**
2. **Conditional Denoising Diffusion Probabilistic Model (C-DDPM)**
3. **Conditional Vision Mamba**

## Key Features

* **Multi-Model Architecture**: Native support for VAE, Diffusion Models, and State Space Models (Mamba) via a Factory pattern.
* **Conditional Generation**: Models accept attribute vectors to guide generation (e.g., *Young*, *Smiling*, *Male*).
* **Advanced Training Management**:
    * Automatic **Checkpointing** system to resume training in case of interruption.
    * Progress monitoring with a custom indicator.
    * Periodic generation of **Visual Snapshots** to qualitatively evaluate learning during epochs.


* **Unified Inference Script**: Flexible tool for generating single samples or comparative grids of all attribute combinations.

## Repository Structure

```text
├── config/             # Configuration and hyperparameter management
├── data/               # Dataset loader and preprocessing (CelebA)
├── models/             # Neural architecture implementations
│   ├── autoencoder/    # Classes for Conditional VAE
│   ├── diffusion/      # Classes for Conditional Diffusion (U-Net, Noise Schedule)
│   └── mamba/          # Classes for Vision Mamba (MambaBlock, PScan)
├── train/              # Training logic (Trainer loop)
├── utility/            # Support tools (Checkpoint, visualization, logging)
├── weights/            # Directory for saving model weights
├── generate.py         # Image generation script
├── train.py            # Training entry point
└── unzip.py            # Utility for dataset extraction

```

## Requirements and Installation

The project requires Python 3.x and standard scientific and deep learning libraries.

1. Clone the repository.
2. Ensure dependencies are installed (PyTorch, Torchvision, NumPy, Matplotlib, PIL).

### Dataset Preparation

The project uses the CelebA dataset. Images must be placed in the `dataset/` folder. A utility script is provided to automatically decompress archives:

```bash
python unzip.py

```

*Note: The script searches for `.zip` files in the `./dataset` folder and extracts contents in place.*

## Configuration

Training configuration is managed via **Environment Variables** or default values defined in `config/config.py`. Parameters can be modified without changing the code by setting variables before execution.

**Main Parameters:**

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_TYPE` | `mamba` | Model type: `vae`, `diff`, `mamba`. |
| `BATCH_SIZE` | `256` | Training batch size. |
| `EPOCHS` | `1000` | Total number of epochs. |
| `LEARNING_RATE` | `5e-4` | Learning rate for the Adam optimizer. |
| `CHECKPOINT_INTERVAL` | `300` | Interval (seconds) for saving checkpoints. |
| `DEVICE` | `cuda` | Computing device (`cuda` or `cpu`). |

## Usage

### Training

To start training, configure the desired model type and run the `train.py` script. The system will automatically load the last available checkpoint if one exists.

Example (Linux/Mac):

```bash
export MODEL_TYPE=vae
export BATCH_SIZE=64
export EPOCHS=50
python train.py

```

Example (Windows PowerShell):

```powershell
$env:MODEL_TYPE="diff"
$env:BATCH_SIZE="32"
python train.py

```

During training, weights will be saved in the `temp/CHECKPOINT` directory, and visual snapshots will be generated to monitor quality.

### Generation (Inference)

The `generate.py` script allows image generation using trained models. It supports two main modes:

1. **Manual Mode**: Generates a set of images with specific attributes.
```bash
python generate.py --model vae --male 1 --smiling 1 --young 1 --num_samples 8 --show

```


2. **Grid Mode (All Combos)**: Generates a grid showing all 8 possible attribute combinations (Male/Female, Smile/NoSmile, Young/Old).
```bash
python generate.py --model diff --all_combos --samples_per_combo 2

```



**Arguments:**

* `--model`: Required. Choice between `vae`, `diff`, `mamba`.
* `--output_dir`: Destination folder (default: `./generated_samples`).
* `--show`: Displays generated images on screen in addition to saving them.

## Model Technical Details

### 1. Conditional VAE

A symmetric Encoder-Decoder architecture with convolutional layers and residual connections (Skip Connections).

* **Encoder**: Compresses the image and attributes into a latent space parameterized by mean ($\mu$) and variance ($\sigma$).
* **Decoder**: Reconstructs the image starting from the sampled latent vector and attributes.

### 2. Conditional Diffusion (DDPM)

Based on a U-Net with sinusoidal *Time Encoding* and attribute conditioning via linear projections.

* Uses a pre-calculated noise schedule (Beta/Alpha schedule).
* Supports guided sampling via the `LAMBDA` ($\lambda$) parameter (Classifier-Free Guidance style).

### 3. Vision Mamba

Innovative implementation based on State Space Models (SSM).

* Utilizes `ResidualMambaLayer` blocks integrating parallel scan operations (`pscan`) for computational efficiency.
* Treats images as sequences of flattened patches, combined with positional and attribute embeddings.

---

**Authors**: Generative AI - Project Group 03 - UNISA A.A. 2025/26
- [Apicella Antonio](https://github.com/apiantonio)
- [Cipriano Ivan](https://github.com/Ivan-02)
- [Faraulo Simone](https://github.com/SimoneFaraulo)
- [Graziosi Antonio](https://github.com/tonygraziosi13)
