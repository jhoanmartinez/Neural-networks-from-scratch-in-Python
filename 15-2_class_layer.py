import numpy as np

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        #self.weights = 0.01 * np.random.rand(n_inputs, n_neurons)
        #self.biases = np.zeros(1, n_neurons)
        pass


    def forward(self, inputs):
        #self.output = np.dot(inputs, self.weights) + self.biases
        pass

n_inputs = 2
n_neurons = 4

weights = 0.01 * np.random.randn(n_inputs, n_neurons)
biases = np.zeros((1, n_neurons))

print("Weights")
print(weights)
print("biases")
print(biases)

On to our forward method — we need to update it with the dot product+biases calculation:
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
