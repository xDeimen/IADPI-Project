# Image Classification using Deep Learning

This project is part of the **MLDAIP** course and explores different deep learning architectures for image classification. The repository contains implementations of various Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs).

## Implemented Models
- **EfficientNet-B0** – A lightweight yet powerful CNN.
- **DenseNet** – A deep network with dense connections.
- **ResNet-50** – A residual network designed for deep learning.
- **Vision Transformer (ViT)** – A transformer-based approach for image classification.

| Feature                | ResNet50              | DenseNet201              | EfficientNet-B0        |
|------------------------|-----------------------|--------------------------|-------------------------|
| **Year Introduced**    | 2015                 | 2017                    | 2019                   |
| **Depth**              | 50 layers            | 201 layers              | Baseline (approx. 82 layers) |
| **Key Innovation**     | Residual connections | Dense connections        | Compound scaling       |
| **Parameters**         | ~25M                 | ~20M                    | ~5.3M                  |
| **Computational Cost** | Moderate             | High                    | Low                    |
| **Accuracy**           | Good                 | Better than ResNet50    | Comparable or better   |
| **Efficiency**         | Moderate             | Moderate to Low          | High                   |



## Dataset
The dataset used for training and evaluation is stored within the respective model directories. It consists of labeled images for supervised learning.

## Training and Evaluation
Each model is trained using PyTorch, with loss and accuracy metrics recorded. The training pipeline includes:
- Data preprocessing with **torchvision**
- Model training with **AdamW optimizer**
- Validation and loss tracking
- Performance evaluation using confusion matrices

## Setup and Usage
1. Install dependencies:
   ```sh
   pip install torch torchvision einops numpy matplotlib seaborn
   ```
2. Run  scripts inside each model directory
3. Evaluate model performance using provided evaluation scripts.


