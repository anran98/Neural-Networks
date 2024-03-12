"""
Programming Assignment:
1. Develop linear, ReLU, sigmoid, tanh, and softmax activation functions as a class for neural networks implementation (This is a graded programming assignment).
2. Develop the class structure and forward propagation including the loss (cost) function implementation for a deep (multilayer) neural network (This is a graded programming assignment)
3. Develop the backpropagation implementation for a deep (multilayer) neural network (This is a mandatory but not graded programming assignment for this time. You must do it to seek a feedback from the TA)
"""

import numpy as np


# Define them as static methods, then we call these methods directly on the class without creating an instance of the class
class ActivationFunctions:

    @staticmethod
    def linear(z):
        return z

    @staticmethod
    def ReLU(z):
        return np.maximum(0, z)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    # Re-scale the input logits into a range where the largest number is 0 and the rest are negative. When do exp(), result in values between 1 and 0
    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(self.weights, inputs) + self.bias
        return self.outputs


class Layer:
    # output_size is the number of neurons in this layer (each neuron will produce one output)
    def __init__(self, input_size, output_size, activation_func):
        # weight vector: np.random.randn(input_size); bias: np.random.randn()
        self.neurons = [
            Neuron(np.random.randn(input_size), np.random.randn())
            for _ in range(output_size)
        ]
        self.activation_func = activation_func
        self.output = None

    def forward(self, inputs):
        # call forward method in the Neuron class
        outputs = np.array([neuron.forward(inputs) for neuron in self.neurons]).reshape(
            -1, 1
        )
        self.output = self.activation_func(outputs)
        return self.output


class ForwardPropagation:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        # the output of each layer becomes the input for the next layer
        for layer in self.layers:
            input = layer.forward(input)
        return input


class BackwardPropagation:
    def __init__(self, network):
        self.network = network

    def backward(self, input, actual):
        # Compute the output of the network (forward propagation)
        predicted = self.network.forward(input)
        m = actual.shape[1]

        loss = self.network.loss_function(predicted, actual)

        for i in reversed(range(len(self.network.layers))):
            layer = self.network.layers[i]

            # Get the input for the current layer
            if i == 0:
                layer_input = input
            else:
                layer_input = self.network.layers[i - 1].output

            # Compute the derivative of the activation function
            if layer.activation_func == ActivationFunctions.sigmoid:
                derivative_activation = layer.output * (1 - layer.output)
            elif layer.activation_func == ActivationFunctions.ReLU:
                derivative_activation = np.where(layer.output > 0, 1, 0)
            elif layer.activation_func == ActivationFunctions.tanh:
                derivative_activation = 1 - np.square(layer.output)
            else:
                derivative_activation = 1

            # Multiply the loss by the derivative of the activation function
            loss *= derivative_activation

            # Compute the gradients
            d_weights = np.dot(layer_input, loss.T) / m
            d_bias = np.sum(loss, axis=1, keepdims=True) / m

            # Update weights and biases of the current layer
            layer.weights -= d_weights.T
            layer.bias -= d_bias

            # Progagate the loss to the previous layer
            if i > 0:
                prev_layer = self.network.layers[i - 1]
                loss = np.dot(prev_layer.weights, loss)


class DeepNeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.forward_propagation = ForwardPropagation(layers)
        self.backward_propagation = BackwardPropagation(self)

    def forward(self, input):
        return self.forward_propagation.forward(input)

    def backward(self, input, actual):
        return self.backward_propagation.backward(input, actual)

    def loss_function(self, A_output, Y):
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A_output + 1e-15))
        return np.squeeze(cost)
