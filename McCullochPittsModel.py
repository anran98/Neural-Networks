"""
Develop software to simulate the McCulloch and Pitts model of artificial neuron. 
The simulated neuron takes input and generates output according to the activation function.           Make sure that your simulation software is designed well to allow for further enhancements.
"""


class McCullochPittsNeuronModel:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def activate(self, inputs):
        # Aggregate by the neuron to get g(x)
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs))

        # The neuron fires when aggregate signal exceeds threshold
        return 1 if weighted_sum >= self.threshold else 0


# Test function
def test_neuron():
    neuron = McCullochPittsNeuronModel(weights=[0.5, 0.5], threshold=1)
    test_cases = [(1, 1), (0, 0)]

    for inputs in test_cases:
        output = neuron.activate(inputs=inputs)
        print(f"Input: {inputs}, Output: {output}")


if __name__ == "__main__":
    test_neuron()
