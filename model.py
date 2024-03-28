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


class Parameters:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros((output_size, 1))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(self.weights, inputs) + self.bias
        return self.outputs


# Randomly mute some neurons in the forward process
class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None

    def apply(self, inputs):
        self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (
            1 - self.rate
        )
        return inputs * self.mask

    # Ensure that the neurons that were dropped out do not contribute to the weight updates
    def backward(self, d_outputs):
        return d_outputs * self.mask


# Modify the loss function: add regularization terms
class Regularization:
    @staticmethod
    def l2_regularization(weights, lambda_val):
        return lambda_val * np.sum(np.square(weights))

    @staticmethod
    def l1_regularization(weights, lambda_val):
        return lambda_val * np.sum(np.abs(weights))


class Normalization:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.mean = None
        self.variance = None

    def forward(self, inputs):
        if self.mean is None:
            self.mean = np.mean(inputs, axis=1, keepdims=True)
            self.variance = np.var(inputs, axis=1, keepdims=True)

        normalized_inputs = (inputs - self.mean) / np.sqrt(self.variance + self.epsilon)
        return normalized_inputs


class Layer:
    # output_size is the number of neurons in this layer (each neuron will produce one output)
    def __init__(
        self,
        input_size,
        output_size,
        activation_func,
        dropout_rate=0,
        regularization_type=None,
        lambda_val=0,
        normalize=False,
    ):
        self.params = Parameters(input_size, output_size)
        self.activation_func = activation_func
        self.dropout = Dropout(dropout_rate) if dropout_rate > 0 else None
        self.regularization_type = regularization_type
        self.lambda_val = lambda_val
        self.normalize = normalize
        if self.normalize:
            self.normalization_layer = Normalization()
        self.output = None
        self.d_weights = None
        self.d_bias = None
        self.reg_cost = 0  # Regularization cost

    def forward(self, inputs):
        if self.dropout:
            inputs = self.dropout.apply(inputs)

        # Compute the outputs
        z = np.dot(self.params.weights, inputs) + self.params.bias
        if self.normalize:
            z = self.normalization_layer.forward(z)
        self.outputs = self.activation_func(z)

        # Apply regularization
        if self.regularization_type == "l2":
            self.reg_cost = Regularization.l2_regularization(
                self.params.weights, self.lambda_val
            )
        elif self.regularization_type == "l1":
            self.reg_cost = Regularization.l1_regularization(
                self.params.weights, self.lambda_val
            )
        else:
            self.reg_cost = 0

        return self.outputs


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

        d_loss = LossFunctions.binary_cross_entropy_loss_derivative(predicted, actual)
        # print(f"Initial d_loss shape: {d_loss.shape}")

        for i in reversed(range(len(self.network.layers))):
            layer = self.network.layers[i]

            # Compute the derivative of the activation function
            if layer.activation_func == ActivationFunctions.sigmoid:
                derivative_activation = layer.outputs * (1 - layer.outputs)
            elif layer.activation_func == ActivationFunctions.ReLU:
                derivative_activation = np.where(layer.outputs > 0, 1, 0)
            elif layer.activation_func == ActivationFunctions.tanh:
                derivative_activation = 1 - np.square(layer.outputs)
            else:
                derivative_activation = 1

            # Multiply the loss by the derivative of the activation function
            d_loss *= derivative_activation

            # Get the input for the current layer
            if i == 0:
                layer_input = input
            else:
                layer_input = self.network.layers[i - 1].outputs

            # Compute the gradients
            layer.d_weights = np.dot(d_loss, layer_input.T) / m
            layer.d_bias = np.sum(d_loss, axis=1, keepdims=True) / m

            # Progagate the loss to the previous layer
            if i > 0:
                # prev_layer = self.network.layers[i - 1]
                d_loss = np.dot(layer.params.weights.T, d_loss)


class GradDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.params.weights -= self.learning_rate * layer.d_weights
        layer.params.bias -= self.learning_rate * layer.d_bias


class LossFunctions:
    # The loss function measures how well the neural network is performing
    @staticmethod
    def binary_cross_entropy_loss(A_output, Y, layers=None):
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A_output) + (1 - Y) * np.log(1 - A_output))

        # Apply regularization cost if provided
        if layers is not None:
            reg_cost = sum(layer.reg_cost for layer in layers)
            cost += reg_cost

        return np.squeeze(cost)

    # The derivative of the loss function is used in backpropagation to update the model's parameters
    @staticmethod
    def binary_cross_entropy_loss_derivative(A_output, Y):
        return -(Y / A_output) + ((1 - Y) / (1 - A_output))


class DeepNeuralNetwork:
    def __init__(self, layer_configs):
        self.layers = []
        self.add_layers(layer_configs)
        self.forward_propagation = ForwardPropagation(self.layers)
        self.backward_propagation = BackwardPropagation(self)

    def add_layers(self, layer_congigs):
        for config in layer_congigs:
            new_layer = Layer(
                input_size=config["input_size"],
                output_size=config["output_size"],
                activation_func=getattr(ActivationFunctions, config["activation_func"]),
                dropout_rate=config.get("dropout_rate", 0),
                regularization_type=config.get("regularization_type", None),
                lambda_val=config.get("lambda_val", 0),
                normalize=config.get("normalize", False),
            )
            self.layers.append(new_layer)

    def forward(self, input):
        return self.forward_propagation.forward(input)

    def backward(self, input, actual):
        return self.backward_propagation.backward(input, actual)

    def loss_function(self, A_output, Y, loss_type="binary_cross_entropy"):
        if loss_type == "binary_cross_entropy":
            return LossFunctions.binary_cross_entropy_loss(A_output, Y)
