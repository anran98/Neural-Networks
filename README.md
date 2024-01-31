import numpy as np

# Activation Class

class Activation:
@staticmethod
def sigmoid(x):
return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

# Neuron Class

class Neuron:
def **init**(self, weights, bias):
self.weights = weights
self.bias = bias

    def activate(self, inputs, activation_func):
        return activation_func(np.dot(inputs, self.weights) + self.bias)

# Layer Class

class Layer:
def **init**(self, num*neurons, num_inputs):
self.neurons = [
Neuron(np.random.randn(num_inputs), 0) for * in range(num_neurons)
]

    def forward(self, inputs, activation_func):
        return np.array(
            [neuron.activate(inputs, activation_func) for neuron in self.neurons]
        ).T

    def update_weights_and_biases(self, dW, db, learning_rate):
        for i, neuron in enumerate(self.neurons):
            neuron.weights -= learning_rate * dW[i]
            neuron.bias -= learning_rate * db[i]

# Model Class

class NeuralNetwork:
def **init**(self):
self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_propagation(self, inputs):
        activations = inputs
        for layer in self.layers:
            activations = layer.forward(activations, Activation.sigmoid)
        return activations

    def back_propagation(self, X, y, output):
        # Backpropagation logic to be implemented
        m = X.shape[1]  # Number of examples
        y = y.reshape(output.shape)  # Reshape y to match output

        dZ = output - y  # Derivative of loss with respect to output
        gradients = []

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            dW = (
                1 / m * np.dot(dZ, self.layers[i - 1].forward(X, Activation.sigmoid).T)
                if i != 0
                else 1 / m * np.dot(dZ, X.T)
            )
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

            if i != 0:
                prev_layer = self.layers[i - 1]
                dA_prev = np.dot(prev_layer.neurons[0].weights, dZ)
                dZ = dA_prev * Activation.sigmoid_derivative(
                    prev_layer.forward(X, Activation.sigmoid)
                )

            gradients.insert(0, (dW, db))

        return gradients

    # Test and evaluate
    def predict(self, X):
        """Make predictions with the trained model"""
        output = self.forward_propagation(X)
        predictions = np.argmax(output, axis=0)
        return predictions

# LossFunction Class

class LossFunction:
@staticmethod
def cross_entropy(predictions, targets): # Clip predictions to avoid log(0) error
predictions_clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
return -np.sum(targets \* np.log(predictions_clipped))

# Training Class

class Training:
def **init**(self, model, learning_rate, target_error):
self.model = model
self.learning_rate = learning_rate
self.target_error = target_error

    def train(self, X, Y):
        # Training loop with forward and backward propagation
        for i in range(self.max_iterations):
            # Forward propagation
            output = self.model.forward_propagation(X)

            # Compute loss
            loss = LossFunction.cross_entropy(output, Y)

            # Check for convergence
            if loss < self.target_error:
                print(f"Training complete at iteration {i}. Loss: {loss}")
                break

            # Backward propagation
            gradients = self.model.back_propagation(X, Y, output)

            # Update weights and biases
            for j, layer in enumerate(self.model.layers):
                dW, db = gradients[j]
                layer.update_weights_and_biases(dW, db, self.learning_rate)

            if i % 100 == 0:
                print(f"Iteration {i}: Loss: {loss}")

# Example Usage

nn_model = NeuralNetwork()
nn_model.add_layer(
Layer(128, 784)
) # Hidden layer with 128 neurons (784 is input size for 28x28 MNIST images)
nn_model.add_layer(Layer(10, 128)) # Output layer with 10 neurons (one for each digit)

# Training logic

trainer = Training(nn_model, learning_rate=0.01, target_error=0.1)

# Add training loop here using trainer.train(X, Y)

# Testing and evaluation logic

# This would include forward passing your test dataset and evaluating the performance

# Evaluation Function

def evaluate_model(model, X_test, y_test):
"""Evaluate the model's performance on the test set"""
predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test) \* 100
return accuracy

# Assuming X_test and y_test are defined and preprocessed similar to your training data

accuracy = evaluate_model(nn_model, X_test, y_test)
print(f"Model Accuracy: {accuracy}%")
