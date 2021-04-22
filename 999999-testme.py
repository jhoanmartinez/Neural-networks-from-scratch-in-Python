import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

data = np.array([
			[5,6],
			[7,4],
					])

y_true = np.array([	[0,0,1],
 					[0,0,1]])

#================================
#	Neural Layers
#================================
class Layer_Dense:

	def __init__(self, n_inputs, n_neurons):
		# self.weights = 0.1*np.random.rand(n_inputs, n_neurons)
		# self.biases = np.zeros((1, n_neurons))
		self.weights = np.array([ [1,2,3], [4,5,6] ])
		self.biases = np.ones( (1, n_neurons) )

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
		self.output = np.maximum(36, input)
		return self.output

#================================
#	Activation Softmax
#================================
class Activation_Softmax:

	def forward_softmax(self, input):
		print("\nSoftmax process")
		exp_values = np.exp(input-np.max(input, axis=1, keepdims=True))
		self.probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
		return self.probabilities

#================================
#	Loss average from cross Entropy
#================================
class Loss:

	#calcula la peridad de datos y regilarizacion
	#valores de salida de verdad
	def calculate(self, output, y):

		#calcula la perdiad de las pmuestras
		sample_losses = self.forward(output, y)

		#calcula la media de la perdida
		data_loss = np.mean(sample_losses)

		#devuelve la media de la perdiad
		return data_loss

#================================
#	Categorical cross Entropy
#================================
class Categorical_Cross_Entropy_Loss(Loss):

	def forward(self, y_pred, y_true):
		samples = len(y_pred)
		y_predict_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

		if len(y_true.shape) == 1:
			correct_confidences = y_predict_clipped[range(samples), y_true]

		elif len(y_true.shape) == 2:
			correct_confidences = np.sum(y_predict_clipped*y_true, axis=1)
			print("\ncorrect confidences = ",correct_confidences)

		log_negatives = -np.log(correct_confidences)
		return log_negatives



#================================
#	Dense 1
#================================
layer_a = Layer_Dense(2, 3)
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

#================================
#	Cross categorical entropy
#================================
entropy = Categorical_Cross_Entropy_Loss()
print("\nCategorical cross entropy\n",entropy.forward(soft_layer_a.probabilities, y_true))

#================================
#	Loss values
#================================
loss = entropy.calculate(soft_layer_a.probabilities, data)
# ToDo{
# 	"1": input, weight, bias,
# 	"2": activation relu,
# 	"3": softmax,
#	"4": categorical cross Entropy
#	"5": Loss
# }
