# Implement input normalization algorithm.

import numpy as np


# Normalize an array of data using z-score normalization.
def normalization(data, epsilon=1e-10):

    # Calculate the mean and standard deviation of the data
    mean = np.mean(data, axis=0)
    var = np.var(data, axis=0)

    # Perform z-score normalization
    normalized_data = (data - mean) / np.sqrt(var + epsilon)

    return normalized_data


# Example usage:
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
normalized_data = normalization(data)
print(normalized_data)
