# Neural Network Implementation

This project contains the implementation of a basic neural network with one hidden layer. The neural network is built from scratch using Python and NumPy, demonstrating fundamental concepts such as forward propagation, backpropagation, and training through gradient descent.

## Features

- Customizable neural network architecture.
- Implementation of the sigmoid activation function.
- Forward propagation for making predictions.
- Backpropagation for training the neural network.
- Gradient descent for optimizing weights and biases.
- Evaluation function for assessing model performance.

## Usage

To use this neural network:

1. **Create an Instance of NeuralNetwork:**
   Initialize the neural network and add layers specifying the number of neurons and inputs.

   ```python
   from neural_network import NeuralNetwork, Layer

   nn_model = NeuralNetwork()
   nn_model.add_layer(Layer(128, 784))  # Example for a hidden layer
   nn_model.add_layer(Layer(10, 128))   # Example for an output layer
   ```
