# McCulloch and Pitts Neuron Model Simulator

## Description

This software simulates the McCulloch and Pitts model of an artificial neuron. It takes input signals, applies weights, and outputs a signal based on an activation threshold.

## Installation

No additional libraries are required for this project. Ensure you have Python 3.x installed on your system.

## Usage

To use the neuron model, create an instance of `McCullochPittsNeuronModel` with desired weights and threshold. Use the `activate` method to simulate neuron activation.

Example:

```python
neuron = McCullochPittsNeuronModel(weights=[0.5, 0.5], threshold=1)
output = neuron.activate(inputs=[1, 1])
print(output)
```
