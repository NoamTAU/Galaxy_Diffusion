# Galaxy Diffusion Models

This repository contains the code and notebooks for training a diffusion model on cosmological galaxy data. The primary goal is to generate realistic galaxy images based on the Flamingo dataset, and subsequently analyze the generated data using forward-backward denoising experiments a la https://arxiv.org/abs/2410.13770

## Quickstart

The main entry point for this project is the `AstroDDPM_Flamingo.ipynb` notebook. It contains all the necessary code for data loading, model training, and generating results.

## Folder Structure

The repository is organized as follows:


├── AstroDDPM_Flamingo.ipynb # <-- MAIN NOTEBOOK: Use this for training and inference.

├── AstroDDPM.ipynb # (Optional) Older or alternative version of the notebook.

├── AstroDDPM.py # (Optional) Python script version.

├── data/ # Directory for storing datasets.

├── checkpoints/ # Directory where trained model weights are saved.

├── results/ # Directory for saving generated images and other outputs.

├── fid_real_images/ # Directory to store real images for FID score calculation.

└── wandb/ # Directory for Weights & Biases logs (created automatically).


-   **`AstroDDPM_Flamingo.ipynb`**: This is the primary Jupyter Notebook you should use. It handles data preprocessing, model definition, the training loop, and visualization.
-   **`data/`**: This folder is intended to hold your training data. You must place your dataset file here.
-   **`checkpoints/`**: During training, model checkpoints will be saved to this directory.
-   **`results/`**: Generated images and other experimental outputs will be saved here.
-   **`wandb/`**: This folder is automatically created by the `wandb` library to store local logs and metadata for your training runs.

## Prerequisites

Before you begin, you need to set up your dataset and environment.

### 1. Dataset

This model is designed to train on the Flamingo dataset, which should be in a `.pkl` (pickle) format.

-   Place your dataset file, `flamingo.pkl`, inside the `data/` directory.
-   In the `AstroDDPM_Flamingo.ipynb` notebook, you **must** update the dataset path variable to point to your file:

```python
# <-- IMPORTANT: Point this to your pickle file
DATASET_PATH = './data/flamingo.pkl'
```

### 2. Weights & Biases (WANDB)

This project uses **Weights & Biases** for experiment tracking, logging, and visualization. You will need a free W&B account to run the training.

-   The training script will prompt you to log in to your W&B account when you run it for the first time.
-   All training metrics, losses, and generated sample images will be logged to your W&B dashboard.

## How to Train

1.  **Set up the dataset** as described in the Prerequisites section.
2.  Open the **`AstroDDPM_Flamingo.ipynb`** notebook in a Jupyter environment.
3.  Follow the instructions and run the cells in order to install dependencies, load the data, and start the training loop.
