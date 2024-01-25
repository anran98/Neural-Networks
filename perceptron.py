"""
1. Prepare a set of grayscale images of handwritten labeled numbers 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 as input for 
training neural networks. The size of the images is 20 x 20. Make 10 different images of each number written 
a little differently.
2. Prepare some test images (not labeled) for testing the trained Perceptron.
3. Develop the software for the Perceptron with input, parameters as transmission matrix and bias including the
module for their initiation (setting their initial values), sigmoid activation function, loss function generation,
forward and backpropagation paths, gradient descent algorithm for training. Make sure that the software allows easy
change of set of images, activation function (various activation functions may be used in the future),
4. Initiate the parameters (transmission matrix and bias).
5. Try to train and test the Perceptron.
"""
import numpy as np


# Neuron class
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias


# Sigmoid Activation class
class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))


# Training class: manages the training process
class Training:
    def __init__(self, neurons, X_train, Y_train, learning_rate, target_error):
        self.neurons = neurons
        self.X_train = X_train
        self.Y_train = Y_train
        self.learning_rate = learning_rate
        self.target_error = target_error

    def propagation(self, neuron, X, Y):
        # How many columns/examples X have
        m = X.shape[1]
        sigmoid = Sigmoid()

        # Forward Propagation
        A = sigmoid.forward(np.dot(neuron.weights.T, X) + neuron.bias)

        # Clip A to avoid log(0)
        A_clipped = np.clip(A, 1e-15, 1 - 1e-15)

        # Cost function: cross-entropy loss function
        cost = (-1 / m) * np.sum(
            Y * np.log(A_clipped) + (1 - Y) * np.log(1 - A_clipped)
        )

        # Backward Propagation
        dw = (1 / m) * np.dot(X, (A - Y).T)
        db = (1 / m) * np.sum(A - Y)

        gradients = {"dw": dw, "db": db}
        return gradients, cost

    def optimize(self):
        all_params = []
        all_costs = []
        for i, neuron in enumerate(self.neurons):
            # Convert to binary labels for each neuron
            Y_train_binary = (self.Y_train == i).astype(int)
            costs = []
            previous_cost = float("inf")
            max_iterations = 1000
            learning_rate_reduction_factor = 0.5

            for _ in range(max_iterations):
                gradients, cost = self.propagation(neuron, self.X_train, Y_train_binary)

                # If training is complete
                if cost < self.target_error:
                    print(f"Training complete for neuron {i}!")
                    break

                # If error is converging
                if cost < previous_cost:
                    previous_cost = cost
                else:
                    self.learning_rate *= learning_rate_reduction_factor
                    print(
                        f"Reducing learning rate for neuron {i} to {self.learning_rate}"
                    )

                dw, db = gradients["dw"], gradients["db"]
                # Update weights and bias
                neuron.weights -= self.learning_rate * dw
                neuron.bias -= self.learning_rate * db
                costs.append(cost)
            all_params.append({"w": neuron.weights, "b": neuron.bias})
            all_costs.append(costs)

            print(f"Training complete for digit {i}. Final cost: {cost}")

        return all_params, all_costs


# Testing Class
class Testing:
    def __init__(self, neurons, X_test, Y_test):
        self.neurons = neurons
        self.X_test = X_test
        self.Y_test = Y_test

    def predict(self, X):
        sigmoid = Sigmoid()
        # Compute the sigmoid activation for each neuron
        predictions = np.array(
            [
                sigmoid.forward(np.dot(neuron.weights.T, X) + neuron.bias)
                for neuron in self.neurons
            ]
        )
        # Choose the digit/class with the highest score
        predicted_class = np.argmax(predictions, axis=0)
        return predicted_class

    def evaluate(self):
        predictions = self.predict(self.X_test)
        accuracy = np.mean(predictions == self.Y_test) * 100
        return accuracy
