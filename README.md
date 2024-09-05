# **EMNIST Digit Recognition using CNN**
This repository contains a Convolutional Neural Network (CNN) implementation for recognizing handwritten digits using the EMNIST dataset. The model is built using TensorFlow and Keras, Aims to achieve high accuracy in recognizing digits from 0 to 9.

 ## Project Overview
Handwritten digit recognition is a common problem in the field of image processing and machine learning. The EMNIST dataset extends the classic MNIST dataset by including both uppercase and lowercase letters. However, in this project, we focus solely on digit recognition.

## Dataset
The dataset used in this project is the EMNIST Digits dataset, which contains 280,000 training images and 40,000 test images of digits. Each image is 28x28 pixels, grayscale.

## Model Architecture
The model is a Convolutional Neural Network (CNN) designed with the following layers:

- Input Layer: 28x28x1 (grayscale image)
- Convolutional Layers: 2D convolutional layers with ReLU activation followed by MaxPooling
- Flattening Layer: Converts the 2D matrix into a vector
- Dense Layers: Fully connected layers with ReLU activation
- Output Layer: A dense layer with softmax activation for 10 classes (digits 0-9)
- The model is compiled with the following parameters:
- Loss Function: Categorical Cross-Entropy
- Optimizer: Adam
- Metrics: Accuracy
- The model is trained for 10 epochs with early stopping, learning rate reduction, and model checkpointing to ensure the best performance.

Evaluation
The model achieves an accuracy of 99 % on the test dataset. The final performance metrics are displayed as a line plot showing the accuracy over epochs.
