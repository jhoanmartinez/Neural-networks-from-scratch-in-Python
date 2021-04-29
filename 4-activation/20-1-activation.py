import numpy as np

X = np.array([
	[3, -6],
	[-3, 5],
	[7, -4],
	[2, -2],
	[8, -2],
	[9, -34]
])

class Layer_Dense:
	def __init__(self, n_inputs, n_neurones):
		self.weights = 0.1 * np.random.randn(n_inputs, n_neurones)
		self.biases = np.zeros((1, n_neurones))

	def forward_layer(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases
		return self.output

class Activation_ReLU:
	def forward_relu(self, input):
		self.acti_out = np.maximum(0, input)
		return self.acti_out

hidden_1 = Layer_Dense(2, 3)
h1 = hidden_1.forward_layer(X)

activation1 = Activation_ReLU()
act1 = activation1.forward_relu(h1)
print(act1[:5])
