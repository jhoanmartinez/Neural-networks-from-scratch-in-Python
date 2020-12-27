import numpy as np

inputs = np.array([
	[-1,-1],
	[-1,4],
])

class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1 * np.random.rand(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))

	def forward(self, inputs):
		print("NO ACTIVATION")
		self.outputs = np.dot(inputs, self.weights) + self.biases
		return self.outputs

hidden_1 = Layer_Dense(2, 3)
print(hidden_1.forward(inputs))

class Activation_ReLU:
	def activate(self, inputs):
		print("\nActivation_ReLU")
		self.acti_out = np.maximum(0, inputs)
		return self.acti_out

acti_hidden = Activation_ReLU()
acti_hidden.activate(hidden_1.outputs)
print(acti_hidden.activate(hidden_1.outputs))
