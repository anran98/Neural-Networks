# Implement a convolution algorithm without padding and without striding for original image 6 x 6 and filter 3 x 3.

import numpy as np


def convolution_2d(image, filter):
    # Get the dimensions of the image and filter
    image_height, image_width = image.shape
    filter_height, filter_width = filter.shape

    # Calculate the dimensions of the output image
    output_height = image_height - filter_height + 1
    output_width = image_width - filter_width + 1

    # Initialize the output image with zeros
    convolved_image = np.zeros((output_height, output_width))

    # Apply the filter to the image
    for i in range(output_height):
        for j in range(output_width):
            # Extract the region of the image
            image_region = image[i : i + filter_height, j : j + filter_width]
            # Perform element-wise multiplication and sum the result
            convolved_image[i, j] = np.sum(image_region * filter)

    return convolved_image


# Example usage
original_image = np.array(
    [
        [3, 0, 1, 2, 7, 4],
        [1, 5, 8, 9, 3, 1],
        [2, 7, 2, 5, 1, 3],
        [0, 1, 3, 1, 7, 8],
        [4, 2, 1, 6, 2, 8],
        [2, 4, 5, 2, 3, 9],
    ]
)

filter = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

convolved_image = convolution_2d(original_image, filter)
print(convolved_image)
