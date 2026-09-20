# Fashion-MNIST Image Classification (PyTorch)

A feedforward neural network built from scratch in PyTorch to classify Fashion-MNIST images into 10 clothing categories.

## Overview

This project trains a simple fully-connected neural network on the Fashion-MNIST dataset — 28×28 grayscale images of clothing items (t-shirts, trousers, shoes, bags, etc.) — using a custom `Dataset`/`DataLoader` pipeline and a manual PyTorch training loop.

## Dataset

- **File:** `fashion-mnist_train.csv`
- **Shape:** 60,000 rows × 785 columns (1 label + 784 pixel values, i.e. flattened 28×28 images)
- **Classes:** 10 (Fashion-MNIST's standard categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

> Note: the dataset is loaded from `/content/fashion-mnist_train.csv` (a Google Colab path). Update this path if running locally.

## Approach

1. **Preprocessing:**
   - 80/20 train/test split (`random_state=42`)
   - Feature scaling with `StandardScaler`
   - Custom PyTorch `Dataset` class wrapping features/labels as tensors
   - `DataLoader` with `batch_size=32` (shuffled for training)
2. **Model architecture** (`MyNN`, fully connected):
   - Input (784) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(10)
3. **Training:**
   - Loss: `CrossEntropyLoss`
   - Optimizer: SGD, `learning_rate=0.1`
   - Epochs: 25
   - Runs on GPU if available (`cuda`), otherwise CPU
4. **Evaluation:** Accuracy on the held-out test set

## Results

| Metric | Value |
|---|---|
| Final training loss (epoch 25) | 0.101 |
| Test Accuracy | 88.15% |

## Requirements

```
torch
pandas
numpy
matplotlib
scikit-learn
```

Install with:
```bash
pip install torch pandas numpy matplotlib scikit-learn
```

## Usage

1. Download the Fashion-MNIST training CSV (e.g. from [Kaggle's Fashion-MNIST CSV dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist)) and place it in your working directory as `fashion-mnist_train.csv` (update the path in the notebook if not using Colab).
2. Run `FMNIST.ipynb` top to bottom.
3. Training runs for 25 epochs; adjust `epochs` and `learning_rate` in the notebook to experiment.

## Project Structure

```
.
├── FMNIST.ipynb                    # Data loading, model, training loop, evaluation
├── fashion-mnist_train.csv         # Dataset (add your own)
└── README.md
```

## Future Improvements

- Replace the fully-connected network with a CNN, which typically performs significantly better on image data
- Add validation-loss tracking per epoch and plot training curves
- Try learning-rate scheduling or a different optimizer (Adam) for faster/better convergence
- Add data augmentation (flips, rotations) to improve generalization
- Save the trained model weights (`torch.save`) for reuse without retraining
