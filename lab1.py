"""
Programming Assignment 
Task 1: Develop a Multilayer Neural Network
Develop a multilayer (deep) neural networks for binary classification problem according to the following specification.
layer 1 with 10 neurons and ReLU activation function
layers 2 and 3 (2 layers shown as power 2 in the spec) with 8 neurons per each layer and ReLU activation function
layer 4 with 4 neurons and ReLU activation function
one (last) layer 5 with one neuron and Sigmoid activation function

Task 2: Develop a traning set
Choose objects to train the network and develop a training set.


"""

import numpy as np
from DNN import Layer, DeepNeuralNetwork, ActivationFunctions, GradDescent
import tensorflow as tf


class MiniBatch:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def generate_batches(self, inputs, outputs):
        assert inputs.shape[1] == outputs.shape[1]
        # Shuffle the dataset:
        indices = np.random.permutation(inputs.shape[1])
        inputs_shuffled = inputs[:, indices]
        outputs_shuffled = outputs[:, indices]

        for i in range(0, inputs.shape[1], self.batch_size):
            inputs_batch = inputs_shuffled[:, i : i + self.batch_size]
            outputs_batch = outputs_shuffled[:, i : i + self.batch_size]
            # Ensure outputs_batch is a 2D array
            if outputs_batch.ndim == 1:
                outputs_batch = outputs_batch.reshape(1, -1)
            yield inputs_batch, outputs_batch


class Training:
    def __init__(self, network, optimizer, num_epochs, batch_size):
        self.network = network
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.mini_batch = MiniBatch(batch_size)

    def train(self, inputs, outputs):
        for epoch in range(self.num_epochs):
            # Mini-batch training
            for inputs_batch, outputs_batch in self.mini_batch.generate_batches(
                inputs, outputs
            ):
                # Forward and backward passes
                predictions = self.network.forward(inputs_batch)
                self.network.backward(inputs_batch, outputs_batch)

                # Update parameters using the GradDescent optimizer
                for layer in network.layers:
                    self.optimizer.update_params(layer)


class ModelEvaluator:
    def __init__(self, network):
        self.network = network

    def evaluate(self, inputs, outputs):
        # Perform a forward pass with the test data
        test_predictions = self.network.forward(inputs)

        # Convert predictions to binary labels with 0.5 as the threshold
        predicted_labels = (test_predictions > 0.5).astype(int)

        # Compute the accuracy (Reshape for proper comparison)
        test_accuracy = np.mean(predicted_labels == outputs.reshape(1, -1))

        return test_accuracy


class DataProcessor:
    def __init__(self, x_train, y_train, x_test, y_test):
        # Normalize pixel values to be between 0 and 1
        self.x_train = x_train / 255.0
        self.x_test = x_test / 255.0

        # Binary classification: let's say we want to classify digits as 'even' (label 0) or 'odd' (label 1)
        self.y_train = (y_train % 2).reshape(1, -1)
        self.y_test = (y_test % 2).reshape(1, -1)

        # Flatten the images for our simple neural network input
        self.x_train = self.x_train.reshape(x_train.shape[0], -1).T
        self.x_test = self.x_test.reshape(x_test.shape[0], -1).T


# Define the network and optimizer
input_size = 784
output_size = 1  # binary classification
batch_size = 64
num_epochs = 100

# Load the training dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Initialize the data processor and preprocess the data
data_processor = DataProcessor(x_train, y_train, x_test, y_test)

layer1 = Layer(
    input_size,
    10,
    ActivationFunctions.ReLU,
    dropout_rate=0.2,
    regularization_type="l2",
    lambda_val=0.001,
    normalize=True,
)
layer2 = Layer(
    10,
    8,
    ActivationFunctions.ReLU,
    dropout_rate=0.2,
    regularization_type="l2",
    lambda_val=0.001,
    normalize=True,
)
layer3 = Layer(
    8,
    8,
    ActivationFunctions.ReLU,
    dropout_rate=0.2,
    regularization_type="l2",
    lambda_val=0.001,
    normalize=True,
)
layer4 = Layer(
    8,
    4,
    ActivationFunctions.ReLU,
    dropout_rate=0.2,
    regularization_type="l2",
    lambda_val=0.001,
    normalize=True,
)
layer5 = Layer(
    4,
    output_size,
    ActivationFunctions.sigmoid,
    dropout_rate=0.2,
    regularization_type="l2",
    lambda_val=0.001,
    normalize=True,
)

# Create the list of layers
layers = [layer1, layer2, layer3, layer4, layer5]
network = DeepNeuralNetwork(layers)

optimizer = GradDescent(learning_rate=0.01)

# Initialize and train the model
trainer = Training(network, optimizer, num_epochs, batch_size)
trainer.train(data_processor.x_train, data_processor.y_train)

# Test the model
evaluator = ModelEvaluator(network)
test_accuracy = evaluator.evaluate(data_processor.x_test, data_processor.y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
