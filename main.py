from perceptron import Neuron, Sigmoid, Training, Testing
import numpy as np
import tensorflow as tf
import scipy.ndimage


# Convert integer labels to one-hot vectors: it allows the model to predict a probability distribution across all classes for each input.
def convert_to_one_hot(labels, num_classes):
    one_hot_encoded = np.eye(num_classes)[labels.reshape(-1)]
    return one_hot_encoded.T


# Prepare data sets
# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Select 10 examples for each digit from 0 to 9 from the training set
examples_per_digit = 10
examples_per_digit_test = 5
selected_images = []
selected_labels = []
test_images = []
test_labels = []

for digit in range(10):
    # Get indices of all examples with the current digit
    indices_of_digit = np.where(y_train == digit)[0]
    indices_of_digit_test = np.where(y_test == digit)[0]

    # Select the first 10 examples for the current digit
    selected_indices = indices_of_digit[:examples_per_digit]
    selected_indices_test = indices_of_digit_test[:examples_per_digit_test]
    selected_images.extend(x_train[selected_indices])
    selected_labels.extend(y_train[selected_indices])
    test_images.extend(x_test[selected_indices_test])
    test_labels.extend(y_test[selected_indices_test])


# Resize images to 20X20
def resize_images(images, size=(20, 20)):
    resized_images = []
    for img in images:
        # Calculate the zoom factor for each dimension
        zoom_factor = (size[0] / img.shape[0], size[1] / img.shape[1])
        # Resize the image using scipy.ndimage.zoom
        resized_image = scipy.ndimage.zoom(
            img, zoom_factor, order=1
        )  # order=1 (bilinear) is typically good for images
        resized_images.append(resized_image)
    return np.array(resized_images)


# Convert the lists to numpy arrays and resize
selected_images = resize_images(selected_images)
test_images = resize_images(test_images)
selected_labels = np.array(selected_labels)
test_labels = np.array(test_labels)

# Convert labels to one-hot encoded format
num_classes = 10  # Number of classes for MNIST
selected_labels_one_hot = convert_to_one_hot(selected_labels, num_classes)
test_labels_one_hot = convert_to_one_hot(test_labels, num_classes)
print(selected_labels_one_hot)
"""
# Visualize all images
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
for i in range(100):  # Displaying 100 images (10 for each digit)
    plt.subplot(10, 10, i + 1)
    plt.imshow(selected_images[i], cmap='gray', interpolation='nearest')
    plt.title(f"Label: {selected_labels[i]}")
    plt.axis('off')
plt.show()
"""

# Reshape images for training and testing
train_images_flatten = selected_images.reshape(selected_images.shape[0], -1).T
test_images_flatten = test_images.reshape(test_images.shape[0], -1).T


# Initalize parameters
def initialParameters(dim):
    w = np.random.randn(dim, 1) * 0.01  # N x 1 matrix
    b = 0
    return w, b


# Create a neuron for each class: for a one-vs-all strategy in multi-class classification. Each neuron specializes in recognizing one digit
num_classes = 10  # Number of classes for MNIST digits
neurons = []
for _ in range(num_classes):
    weights, bias = initialParameters(train_images_flatten.shape[0])
    neurons.append(Neuron(weights, bias))


# Train the perceptron
learning_rate, target_error = 0.03, 0.1
model = Training(
    neurons, train_images_flatten, selected_labels, learning_rate, target_error
)
model_params, model_costs = model.optimize()

# Test the model
tester = Testing(neurons, test_images_flatten, test_labels)
accuracy = tester.evaluate()
print(f"Test Accuracy: {accuracy}%")
