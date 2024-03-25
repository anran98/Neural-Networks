# Develop a program for depthwise and pointwise convolutions. The program should take an initial image, kernel (filter), and produce the convoluted image based on the external flag that indicates either the depthwise or pointwise convolution.

import numpy as np
from PIL import Image


def depthwise_convolution(image, kernel):
    # Number of channels in image and kernel must match
    assert image.shape[-1] == kernel.shape[-1]
    image_height, image_width, channels = image.shape
    kernel_height, kernel_width, _ = kernel.shape

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    # Initialize the output image with zeros
    output = np.zeros((output_height, output_width, channels))

    # Apply the filter to the image
    for c in range(channels):
        for i in range(output_height):
            for j in range(output_width):
                # Extract the region of the image
                image_region = image[i : i + kernel_height, j : j + kernel_width, c]
                # Perform element-wise multiplication and sum the result
                output[i, j, c] = np.sum(image_region * kernel[:, :, c])

    return output


def pointwise_convolution(image, kernel):
    image_height, image_width, channels = image.shape
    output_channels = kernel.shape[0]
    # The second dimension of the kernel must match the number of channels in the image
    assert kernel.shape[1] == channels

    output = np.zeros((image_height, image_width, output_channels))

    for i in range(image_height):
        for j in range(image_width):
            image_region = image[i, j, :]
            output[i, j, :] = np.dot(kernel, image_region)

    return output


def main(image_path, kernel, flag="None"):
    image = Image.open(image_path)
    image = np.array(image)

    # Convert grayscale images to 3D arrays
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    if flag == "depthwise":
        convoluted_image = depthwise_convolution(image, kernel)
    elif flag == "pointwise":
        convoluted_image = pointwise_convolution(image, kernel)
    else:
        raise ValueError("Invalid flag: Use 'depthwise' or 'pointwise'.")

    # Convert the result to a PIL image and display it
    convoluted_image = np.clip(convoluted_image, 0, 255).astype(np.uint8)
    result = Image.fromarray(convoluted_image)
    result.show()
