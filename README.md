# Perceptron Neural Network for MNIST Digit Recognition

This project involves the implementation of a perceptron neural network to recognize handwritten digits from the MNIST dataset.

## Files in the Project

### `main.py`

- Loads and processes the MNIST dataset.
- Normalizes pixel values and resizes images to 20x20.
- Initializes the perceptron model with the necessary parameters.
- Handles the training and testing of the perceptron model.

### `perceptron.py`

- Contains the implementation of the perceptron, including classes for InputData, Neuron, Sigmoid, Training, and Testing.
- Manages the forward and backward propagation, cost function calculation, and optimization of the model.
- Evaluates the model's performance on the test data.

## How to Run

1. Ensure you have Python installed on your machine.
2. Install necessary libraries: `numpy`, `tensorflow`, and `scipy`.
3. Clone this repository to your local machine.
4. Navigate to the directory containing `main.py`.
5. Run the script: `python main.py`.

## Additional Information

- The `main.py` script selects a subset of the MNIST dataset for training and testing purposes.
- The perceptron model uses a sigmoid activation function and cross-entropy loss for binary classification of each digit.
- The accuracy of the model is printed out at the end of the testing phase.
