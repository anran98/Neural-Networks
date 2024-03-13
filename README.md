# Multilayer Neural Network for MNIST Classification

This project implements a multilayer neural network (deep learning model) to classify handwritten digits from the MNIST dataset. The model is built using NumPy and TensorFlow and includes features like dropout, L2 regularization, and normalization to improve generalization and prevent overfitting.

## Features

- Multilayer neural network architecture with customizable layers and neurons.
- Activation functions: ReLU, Sigmoid, Tanh, and Softmax.
- Dropout for regularization and to prevent overfitting.
- L2 regularization for weights.
- Batch normalization for faster convergence and better performance.
- Mini-batch gradient descent for training optimization.

## Requirements

- Python 3.x
- NumPy
- TensorFlow

## Architecture

The neural network architecture is defined as follows:

- Layer 1: 10 neurons, ReLU activation
- Layer 2: 8 neurons, ReLU activation
- Layer 3: 8 neurons, ReLU activation
- Layer 4: 4 neurons, ReLU activation
- Output Layer: 1 neuron, Sigmoid activation
