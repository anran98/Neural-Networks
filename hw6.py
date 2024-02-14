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


class DeepNeuralNetwork:
    def __init__(self, layer_dims):
        self.parameters = self.initialize_parameters(layer_dims)

    def initialize_parameters(self, layer_dims):
        parameters = {}
        for i in range(1, len(layer_dims)):
            parameters["W" + str(i)] = (
                np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
            )
            parameters["b" + str(i)] = np.zeros((layer_dims[i], 1))
        return parameters

    def forword_propagation(self, X):
        forward = {"A0": X}
        # There are W and b in parameters, total is 2L
        final_layer = len(self.parameters) // 2

        for l in range(1, final_layer + 1):
            forward["Z" + str(l)] = (
                np.dot(self.parameters["W" + str(l)], forward["A" + str(l - 1)])
                + self.parameters["b" + str(l)]
            )

            # Output layer: use softmax function
            if l == final_layer:
                forward["A" + str(l)] = ActivationFunctions.softmax(
                    forward["Z" + str(l)]
                )
            # Hidden layers: use ReLU function
            else:
                forward["A" + str(l)] = ActivationFunctions.ReLU(forward["Z"] + str(l))

        return forward["A" + str(final_layer)], forward

    def loss_function(self, A_output, Y):
        m = Y.shape[1]
        E = np.ones(Y.shape)
        cost = -(1 / m) * np.sum(
            Y * np.log(A_output.T) + (E - Y) * np.log((E - A_output).T)
        )
        return cost

    def backward_propagation(self, Y, forward):
        grads = {}
        L = len(self.parameters) // 2
        m = Y.shape[1]

        # Gradients for the output layer L
        AL = forward["A" + str(L)]
        grads["dZ" + str(L)] = AL - Y
        grads["dW" + str(L)] = (
            np.dot(grads["dZ" + str(L)], forward["A" + str(L - 1)].T) / m
        )
        grads["db" + str(L)] = np.sum(grads["dZ" + str(L)], axis=1, keepdims=True) / m

        # Gradients for the hidden layers L-1 down to 1
        for l in reversed(range(1, L)):
            dZ_next = grads["dZ" + str(l + 1)]
            W_next = self.parameters["W" + str(l + 1)]
            Z_curr = forward["Z" + str(l)]
            A_prev = forward["A" + str(l - 1)]

            grads["dA" + str(l)] = np.dot(W_next.T, dZ_next)
            grads["dZ" + str(l)] = grads["dA" + str(l)] * (Z_curr > 0)
            grads["dW" + str(l)] = np.dot(grads["dZ" + str(l)], A_prev.T) / m
            grads["db" + str(l)] = (
                np.sum(grads["dZ" + str(l)], axis=1, keepdims=True) / m
            )

        return grads

    def update_parameters(self, grads, learning_rate):
        final_layer = len(self.parameters) // 2  # number of layers

        for l in range(1, final_layer + 1):
            self.parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
            self.parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
