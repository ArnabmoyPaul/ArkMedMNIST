# ArkMedMNIST

ArkMedMNIST is a PyTorch-based framework implementing a teacher-student cyclic training approach (Ark+) for medical image classification using the MedMNIST v2 collection. The project supports both 2D and 3D medical datasets.

## Features

- **Teacher-Student Architecture:** Implements a consistency-based teacher-student model with Exponential Moving Average (EMA) parameter updates.
- **Cyclic Pretraining:** Trains a single encoder across multiple datasets in a cyclic manner using dataset-specific classification heads ("omni heads").
- **2D Support (12 Datasets):** Utilizes a Swin Transformer backbone (`swin_base_patch4_window7_224` from `timm`) with support for multi-class and multi-label classification tasks. 
- **3D Support (3 Datasets):** Uses a custom 3D ResNet-based backbone and includes Automatic Mixed Precision (AMP) for efficient training.
- **Loss Formulation:** Combines classification loss (Cross-Entropy/BCE) on the student's predictions with a Mean Squared Error (MSE) consistency loss between the teacher's and student's representations.

## Supported Datasets

**2D Datasets (12):**
- PathMNIST, BloodMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST
- BreastMNIST, TissueMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST, ChestMNIST

**3D Datasets (3):**
- OrganMNIST3D, NoduleMNIST3D, FractureMNIST3D

## Repository Structure

- `train_ark_12datasets.py` & `train_ark_12datasets.ipynb`: Training scripts for 2D datasets.
- `train_ark_3d_3datasets.py` & `train_ark_3d_3datasets.ipynb`: Training scripts for 3D datasets.
- `ark_plus_model.py`, `models.py`, `convnext.py`: Model definitions and backbones.
- `dataloader.py`, `medmnist_dataloader.py`, `medmnist3d_dataloader.py`: Custom PyTorch dataset wrappers for MedMNIST.
- `engine.py`, `trainer.py`, `utils.py`: Training engine, cyclic training logic, and utility functions.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- PyTorch
- torchvision
- timm
- scikit-learn
- numpy
- medmnist (can be installed via `pip install medmnist`)

### Running the Code

To train the 2D model across all 12 datasets:
```bash
python train_ark_12datasets.py
```

To train the 3D model:
```bash
python train_ark_3d_3datasets.py
```

Outputs, including training logs and best model checkpoints, will be saved in the `./outputs/` directory.
