import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense:
	def __init__(self, n_inputs, n_neurones):
		self.weights = 0.1 * np.random.rand(n_inputs, n_neurones)
		self.biases = np.zeros((1, n_neurones))

	def forward_1(self, inputs):
		self.outputs = np.dot(inputs, self.weights) + self.biases
		return self.outputs

class Activation_ReLU:
	def forward_2(self, inputs):
            self.outputs = np.maximum(0, inputs)

X, y = spiral_data(samples=100, classes=3)
print("Input SHAPE\n",X.shape,"\n")

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Make a forward pass of our training data through this layer
dense1.forward_1(X)

# Forward pass through activation func.
# Takes in output from previous layer
activation1.forward_2(dense1.outputs)

# Let's see output of the first few samples:
print(activation1.outputs[:5])

"""testing nvim """
