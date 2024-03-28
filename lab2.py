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
from model import DeepNeuralNetwork, GradDescent
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
    def __init__(
        self,
        network,
        optimizer,
        num_epochs,
        batch_size,
        target_error,
        learning_rate_reduction_factor,
        patience=10,
    ):
        self.network = network
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.mini_batch = MiniBatch(batch_size)
        self.target_error = target_error
        self.learning_rate_reduction_factor = learning_rate_reduction_factor
        self.patience = patience  # Number of epochs to wait before early stopping
        self.best_val_error = float("inf")
        self.best_model_params = []
        self.epochs_without_improvement = 0

    def train(self, inputs, outputs, validation_data=None):

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

            # Validation and early stopping
            if validation_data is not None:
                val_inputs, val_outputs = validation_data
                val_predictions = self.network.forward(val_inputs)
                val_error = np.mean(np.abs(val_outputs - val_predictions))

                # Check if the error is less than the target error
                if val_error < self.target_error:
                    print(
                        f"Target error of {self.target_error} reached at epoch {epoch} with validation error {val_error}"
                    )
                    return

                # Early stopping and learning rate adjustment
                if val_error < self.best_val_error:
                    self.best_val_error = val_error
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                    if self.epochs_without_improvement >= self.patience:
                        print(
                            f"Early stopping at epoch {epoch} with validation error {self.best_val_error}"
                        )
                        return
                    elif self.epochs_without_improvement % self.patience == 0:
                        self.optimizer.learning_rate *= (
                            self.learning_rate_reduction_factor
                        )
                        print(
                            f"Reducing learning rate to {self.optimizer.learning_rate} and repeating training"
                        )

        print(f"Training completed after {epoch + 1} epochs")


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


# Load the training dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Initialize the data processor and preprocess the data
data_processor = DataProcessor(x_train, y_train, x_test, y_test)


input_size = 784
output_size = 1  # binary classification

# Define the configurations for each layer
layer_configs = [
    {
        "input_size": input_size,
        "output_size": 10,
        "activation_func": "ReLU",
        "dropout_rate": 0.2,
        "regularization_type": "l2",
        "lambda_val": 0.001,
        "normalize": True,
    },
    {
        "input_size": 10,
        "output_size": 8,
        "activation_func": "ReLU",
        "dropout_rate": 0.2,
        "regularization_type": "l2",
        "lambda_val": 0.001,
        "normalize": True,
    },
    {
        "input_size": 8,
        "output_size": 8,
        "activation_func": "ReLU",
        "dropout_rate": 0.2,
        "regularization_type": "l2",
        "lambda_val": 0.001,
        "normalize": True,
    },
    {
        "input_size": 8,
        "output_size": 4,
        "activation_func": "ReLU",
        "dropout_rate": 0.2,
        "regularization_type": "l2",
        "lambda_val": 0.001,
        "normalize": True,
    },
    {
        "input_size": 4,
        "output_size": output_size,
        "activation_func": "sigmoid",
        "dropout_rate": 0.2,
        "regularization_type": "l2",
        "lambda_val": 0.001,
        "normalize": True,
    },
]


# Create the network using the configurations
network = DeepNeuralNetwork(layer_configs)

# Define other training parameters
batch_size = 64
num_epochs = 100
target_error = 0.05  # Example target error
learning_rate_reduction_factor = 0.5  # Example learning rate reduction factor
patience = 5  # Example patience for early stopping
optimizer = GradDescent(learning_rate=0.01)

# Initialize the training class and train the model
trainer = Training(
    network=network,
    optimizer=optimizer,
    num_epochs=num_epochs,
    batch_size=batch_size,
    target_error=target_error,
    learning_rate_reduction_factor=learning_rate_reduction_factor,
    patience=patience,
)
trainer.train(data_processor.x_train, data_processor.y_train)

# Test the model
evaluator = ModelEvaluator(network)
test_accuracy = evaluator.evaluate(data_processor.x_test, data_processor.y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
