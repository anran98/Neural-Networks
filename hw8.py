"""Programming Assignment
Develop a prototype of the training, validation, and testing sets of your choice for the future training of your neural network. 
The term “prototype: means that the sets may be quite limited by number of images, but the proportions of images in them should be maintained as required
"""

# Import necessary libraries
from sklearn.model_selection import train_test_split


# Function to split dataset into train, validation, and test sets
def split_dataset(dataset, model_complexity="medium"):
    data_size = len(dataset)

    # Define split ratios based on data size
    if data_size <= 1000:
        split_ratios = {"train": 0.7, "validation": 0.15, "test": 0.15}
    elif 10000 <= data_size <= 100000:
        split_ratios = {"train": 0.9, "validation": 0.05, "test": 0.05}
    elif data_size > 1000000:
        split_ratios = {"train": 0.98, "validation": 0.01, "test": 0.01}

    # Split the dataset into training and temporary sets
    train_data, temp_data = train_test_split(
        dataset,
        test_size=(split_ratios["validation"] + split_ratios["test"]),
        random_state=42,
    )

    # Split the temporary set into validation and test sets
    validation_data, test_data = train_test_split(
        temp_data,
        test_size=split_ratios["test"]
        / (split_ratios["validation"] + split_ratios["test"]),
        random_state=42,
    )

    return train_data, validation_data, test_data


# Example usage with a prototype dataset:
# For the purpose of this example, let's assume we have 100 data points (e.g., image paths).
prototype_dataset = ["data_{}".format(i) for i in range(1, 101)]

# Perform the split
train_data, validation_data, test_data = split_dataset(prototype_dataset)

# Get the size of each dataset to verify the proportions
(train_data_size, val_data_size, test_data_size) = (
    len(train_data),
    len(validation_data),
    len(test_data),
)

print(train_data_size, val_data_size, test_data_size)
