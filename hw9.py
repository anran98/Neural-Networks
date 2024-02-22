# Implement regularization algorithms in your neural network.
# Implement dropout algorithms in your neural network.

import numpy as np


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

    # Implement Dropout
    def forword_propagation_with_dropout(self, X, keep_prob=0.8):
        forward = {"A0": X}
        # There are W and b in parameters, total is 2L
        final_layer = len(self.parameters) // 2

        # Dropout masks
        forward["D"] = {}

        # Apply Dropout
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
                # Apply dropout only to the hidden layers and if keep_prob is less than 1
                if keep_prob < 1:
                    forward["D"][str(l)] = (
                        np.random.rand(
                            forward["A" + str(l)].shape[0],
                            forward["A" + str(l)].shape[1],
                        )
                        < keep_prob
                    )
                    forward["A" + str(l)] = (
                        forward["A" + str(l)] * forward["D"][str(l)]
                    )  # Apply mask
                    # Scale the values of the neurons that haven't been dropped
                    forward["A" + str(l)] /= keep_prob

        return forward["A" + str(final_layer)], forward

    # Implement Dropout
    def backward_propagation_with_dropout(self, Y, forward, keep_prob=0.8):
        grads = {}
        L = len(self.parameters) // 2
        m = Y.shape[1]

        # Gradients for the output layer L
        AL = forward["A" + str(L)]
        grads["dA" + str(L)] = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        for l in reversed(range(1, L + 1)):
            dA = grads["dA" + str(l)]
            # Apply dropout mask to gradients for hidden layers
            if l < L and keep_prob < 1:
                dA = dA * forward["D"][str(l)]
                dA /= keep_prob

            dZ = dA * ActivationFunctions.ReLU_deriv(forward["Z" + str(l)])
            grads["dW" + str(l)] = 1 / m * np.dot(dZ, forward["A" + str(l - 1)].T)
            grads["db" + str(l)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                grads["dA" + str(l - 1)] = np.dot(self.parameters["W" + str(l)].T, dZ)

        return grads

    # Implement L2 regularization
    def loss_function_with_regularization(self, A_output, Y, lambda_reg=0.7):
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A_output.T) + (1 - Y) * np.log((1 - A_output).T)
        )

        L2_cost = 0
        L = len(self.parameters) // 2  # number of layers
        for l in range(1, L + 1):
            L2_cost += np.sum(np.square(self.parameters["W" + str(l)]))

        L2_cost = (lambda_reg / (2 * m)) * L2_cost
        cost += L2_cost

        return cost

    def update_parameters(self, grads, learning_rate):
        final_layer = len(self.parameters) // 2  # number of layers

        for l in range(1, final_layer + 1):
            self.parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
            self.parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]


class ActivationFunctions:
    @staticmethod
    def ReLU(z):
        return np.maximum(0, z)

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
