# Mini-Batch Implementation

## Overview
This Python implementation demonstrates the mini-batch approach commonly used in training machine learning models. By processing data in small batches, this method can improve computational efficiency and model convergence.

## Features
- **Batch Creation:** Generates mini-batches from the input dataset.
- **Data Shuffling:** Randomly shuffles the data before batching to ensure variability.
- **Last Batch Handling:** Handles the final batch, which may contain fewer samples than the specified batch size.

## Usage
To use this implementation, you need to have `numpy` installed. You can create mini-batches from your data as follows:

```python
import numpy as np
from mini_batch import create_mini_batches

