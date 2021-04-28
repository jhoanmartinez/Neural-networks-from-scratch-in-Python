import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# X = np.array([
# 				[5,6],
# 				[7,4],
# 						])
# #
# y = np.array([0,1])

#================================
#	 Dense Layer
#================================
class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.01 * np.random.rand(n_inputs, n_neurons)
		self.biases = np.zeros( (1, n_neurons) )

	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases

#================================
#	 ReLU activation
#================================
class Activation_ReLU:
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)

#================================
#	Softmax activation
#================================
class Activation_Softmax:
	def forward(self, inputs):
		exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities

#================================
#	2. Loss mean values
#================================
class Loss:
	def calculate(self, a, b):
		output = self.forward(a, b)
		mean_error = np.mean(output)
		self.output = mean_error

#================================
#	1. Categorical cross Entropy
#================================
class Loss_CategoricalCrossentropy(Loss):
	def forward(self, y_pred, clases):
		samples = len(y_pred) #300 filas
		y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) #evitar q sea cero y mas de 0.9
		if len(clases.shape) == 1:
			values = y_pred_clipped[range(samples), clases]
		elif len(clases.shape) == 2:
			values = np.sum(y_pred_clipped*clases, axis=1)
		neg_log = -np.log(values)
		return neg_log
		# self.mean_error = np.mean(-np.log(values)) =====> Esperando respuesta

#================================
#	Dataset
#================================
X = spiral_data(samples=100, classes=3)[0][:2] #shape = n filas y 2 columnas
y = spiral_data(samples=100, classes=3)[1][:2] #shape = n filas

#================================
#	Iniciar objetos de la red
#================================
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss = Loss_CategoricalCrossentropy()

#================================
#		proceso de la red
#================================
dense1.forward(X)
print("inputs*weights+biases\n",dense1.output)
activation1.forward(dense1.output)
print("ReLU\n",activation1.output)

dense2.forward(activation1.output)
print("inputs*weights+biases\n", dense2.output)
activation2.forward(dense2.output)
print("softmax\n", activation2.output)

loss.calculate(activation2.output, y)
print("loss\n", loss.output)
# loss.forward(activation2.output, y) ====> Esperando respuesta
# print("loss\n", loss.mean_error) =======> Esperadno respuesta

# ToDo{
# 	"1": input, weight, bias,
# 	"2": activation relu,
# 	"3": softmax,
#	"4": categorical cross Entropy
#	"5": Loss
# }
