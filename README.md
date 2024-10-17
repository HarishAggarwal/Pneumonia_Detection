# Pneumonia Detection using Convolutional Neural Networks (CNN)

This project aims to detect pneumonia from chest X-ray images using a Convolutional Neural Network (CNN). The model is designed to classify X-ray images into two categories: "Pneumonia" and "Normal." The approach leverages data augmentation and deep learning techniques to create an accurate model that can assist in the early detection of pneumonia.

## Dataset

The dataset used consists of chest X-ray images divided into two classes:
- **Pneumonia**: X-rays of patients diagnosed with pneumonia.
- **Normal**: X-rays of healthy individuals.

The dataset is split into:
- **train/**: Used for training the model.
- **val/**: Used for validating the model during training.

The dataset is available from [Kaggle's Chest X-ray Pneumonia dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

## Model Overview

The CNN model used in this project consists of multiple convolutional layers, followed by max-pooling and dense layers. It is trained using binary cross-entropy as the loss function and the Adam optimizer. The output layer contains a single neuron with a sigmoid activation function for binary classification.

### Key Features:
- **Data Augmentation**: Used to prevent overfitting by applying transformations like rotation, zoom, shift, and flips.
- **Batch Size**: Training images are processed in batches of 32.
- **Input Size**: All X-ray images are resized to 224x224 pixels.
- **Training Epochs**: The model is trained for 10 epochs.

## Results

The model's performance is evaluated using accuracy and loss metrics on both the training and validation datasets. After training, accuracy and loss plots are generated to assess performance.

## Dependencies

- TensorFlow
- Keras
- NumPy
- Matplotlib

## How to Run

1. Download and organize the dataset.
2. Install required dependencies.
3. Run the `CNN.py` script to train the model.
