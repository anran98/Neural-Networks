# Implement a mini batch approach.

import numpy as np


"""
Args:
    X (np.array): Input features of shape (num_examples, num_features)
    y (np.array): Output labels of shape (num_examples, 1)
    batch_size (int): The size of each mini-batch

Returns:
    list: A list of tuples (mini_batch_X, mini_batch_y)
"""


def create_mini_batches(X, y, batch_size):
    mini_batches = []
    data = np.hstack((X, y))
    np.random.shuffle(data)
    # number of complete mini-batches
    n_minibatches = data.shape[0] // batch_size

    for i in range(n_minibatches):
        mini_batch = data[i * batch_size : (i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))  # reshape into column vector
        mini_batches.append((X_mini, Y_mini))

    # Remaining examples that cannot fit into a complete mini-batch
    if data.shape[0] % batch_size != 0:
        mini_batch = data[n_minibatches * batch_size : data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))

    return mini_batches


# Example usage:
# X_train = np.random.randn(1024, 100)  # 1024 samples, 100 features
# y_train = np.random.randn(1024, 1)    # 1024 labels
# batch_size = 64
# mini_batches = create_mini_batches(X_train, y_train, batch_size)
