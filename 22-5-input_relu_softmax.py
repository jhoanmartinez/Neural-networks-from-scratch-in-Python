import numpy as np

inputs = np.array([
	[1,2],
	[1,2],
	[1,2],
	[1,2],
	[1,2],
	[1,2],
	[1,2],
])


"""Capa oculta que recibe los inputs"""
class Layer_Dense:

	def __init__(self, inputs, n_neurons):
		self.weights = 0.1 * np.random.rand(inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))

	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases


"""Funcion de activacion ReLU"""
class Activation_ReLU():

	def forward(self, inputs):
		self.output = np.maximum(0, inputs)


"""Softmax class"""
class Activation_Softmax:

	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilites = exp_values / np.sum(exp_values)
		self.output = probabilites

"""Intriduciendo datos"""

"""Creo los objetos que voy a implementar"""
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

"""Paso los parametros a los objetos creados anteriormente"""
# Make a forward pass of our training data through this layer
dense1.forward(inputs)

# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)

# Let's see output of the first few samples:
print(activation2.output[:5])
