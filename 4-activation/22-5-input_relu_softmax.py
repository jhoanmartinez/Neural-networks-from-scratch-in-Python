import numpy as np



"""Capa oculta que recibe los inputs"""
class Layer_Dense:

	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1 * np.random.rand(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))

	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases

"""Capa de activacion ReLU """
class Activation_ReLU():

	def forward(self, inputs):
		self.output = np.maximum(0, inputs)


"""Activation Softmax"""
class Activation_Softmax:

	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities

inputs = np.array([
	[1,2,3],
	[4,5,6],
])

dense1 = Layer_Dense(3, 2)
dense1.forward(inputs)
print("\nDense inputs 1\n",inputs)
print("\nDense weights 1\n",dense1.weights)
print("\nDense biases 1\n",dense1.biases)
print("\nDense output 1\n",dense1.output)


relu1 = Activation_ReLU()
relu1.forward(dense1.output)
print("\nReLU 1\n",relu1.output)

# dense2 = Layer_Dense(3, 2)
# dense2.forward(relu1.output)
# print("\nDense 2\n",dense2.output)
#
# relu2 = Activation_ReLU()
# relu2.forward(dense2.output)
# print("\nReLU 2\n",relu2.output)
#
# dense3 = Layer_Dense(2, 3)
# dense3.forward(relu2.output)
# print("\nDense 3\n",dense3.output)
#
# relu3 = Activation_ReLU()
# relu3.forward(dense3.output)
# print("\nReLU 3\n",relu3.output)

soft1 = Activation_Softmax()
soft1.forward(relu1.output)
print("\nSoftmax\n",soft1.output)
