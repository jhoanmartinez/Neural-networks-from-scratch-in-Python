
import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

X = np.array([
    [4,5],
    [2,3],
    [7,6],
    [1,6]
])

hidden_1 = Layer_Dense(2, 3)
hidden_1.forward(X)
print(hidden_1.output[:5])
