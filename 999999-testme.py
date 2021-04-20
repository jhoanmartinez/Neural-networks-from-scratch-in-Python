import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

data = np.array(
		[
			[1,2,3],
			[4,5,6],
		]
)

#================================
#	Neural Layers
#================================
class Layer_Dense:

	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1*np.random.rand(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))

	def forward(self, inputs):
		print("\n Hidden Layer output")
		self.output = np.dot(inputs, self.weights) + self.biases
		return self.output

#================================
#	Activaction ReLU
#================================
class Activation_ReLU:

	def forward_relu(self, input):
		print("\n Activation ReLU")
		self.output = np.maximum(0, input)
		return self.output

#================================
#	Activation Softmax
#================================
class Activation_Softmax:

	def forward_softmax(self, input):
		print("\nSoftmax process")
		exp = np.exp(input)
		sum = np.sum(exp)
		self.probabilities = exp/sum
		return self.probabilities

#================================
#	Dense 1
#================================
layer_a = Layer_Dense(3, 3)
print(layer_a.forward(data))

#================================
#	activation ReLU 1
#================================
act_layer_a = Activation_ReLU()
print(act_layer_a.forward_relu(layer_a.output))

## Ense 2 => ReLU 2 , Dense 3 => ReLU 3 ...

#================================
#	activation softmax al final
#================================
soft_layer_a = Activation_Softmax()
print(soft_layer_a.forward_softmax(act_layer_a.output))

# ToDo{
# 	"1": {input, weight, bias},
# 	"2": {activation relu},
# 	"3": {softmax},
# }
