import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# X = np.array([
# 	[0.7, 0.1],
# 	[0.1, 0.5],
# 	[0.1, 0.9]
# ])
#
# y = np.array([0, 1, 1])

# Perform a forward pass through loss function
# it takes the output of second dense layer here and returns loss


#layer dense hidden
class Layer_Dense:

	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1*np.random.rand(n_inputs, n_neurons)
		self.biases = np.zeros( (1, n_neurons) )

	def forward(self, inputs):
		self.outputs = np.dot(inputs, self.weightsw) + self.biases
		return self.outputs

# ReLU activation
class Activation_ReLU:

	def forward(self, input):
		self.output = np.maximum(0, input)

# Activation softmax
class Activation_softmax:

	def forward(self, inputs):
		exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities

#Loss y categorical cross Entropy
class Loss_Categorical_Cross_Entropy:

	def forward(self, y_pred, y_true):
		samples = len(y_pred) # 300 filas de datos
		y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) #< a 0 sube a 0.0001 > a 1 baja a 0.999

		if y_true.shape == 1:
			correct_confidences = y_pred_clipped[range(samples), y_true]

		elif y_true.shape == 2:
			correct_confidences = np.sum(np.dot(y_pred_clipped, y_true), axis=1)

# Create dataset
X, y = spiral_data(samples=100, classes=3)
