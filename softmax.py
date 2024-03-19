# Implement the softmax activation function

import numpy as np


def softmax(z):
    # Subtract the max for numerical stability
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    # Calculate the softmax
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


# Example usage:
z = np.array([2.0, 1.0, 0.1])
print(softmax(z))
