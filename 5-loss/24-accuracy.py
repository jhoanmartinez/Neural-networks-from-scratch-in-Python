import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data
nnfs.init()

#================================
#	 Dense Layer
#================================
class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
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
		return mean_error

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
#	Accuracy
#================================
class Accuracy:
	def forward(self, predictions, clas):
		pred = np.argmax(predictions, axis = 1)
		if len(clas.shape) == 2:
			clas = np.argmax(clas, axis = 1)
		self.accuracy = np.mean(pred == clas)
		return self.accuracy
#================================
#	Dataset
#================================
# X, y = spiral_data(samples=100, classes=3) #[:5] #shape = n filas y 2 columnas
X, y = vertical_data(samples=100, classes=3) #[:5] #shape = n filas y 2 columnas

#================================
#	Iniciar objetos de la red
#================================
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss = Loss_CategoricalCrossentropy()

accuracy = Accuracy()

#================================
#		proceso de la red
#================================
# dense1.forward(X)
# print("weights",dense1.weights)
# print("biases",dense1.biases)

# activation1.forward(dense1.output)
# print("ReLU\n",activation1.output)

# dense2.forward(activation1.output)
# print("inputs*weights+biases\n", dense2.output)
# activation2.forward(dense2.output)
# print("softmax\n", activation2.output[:5])

# loss.calculate(activation2.output, y)

# accuracy.forward(activation2.output, y)
# print("Accuracy:",accuracy.accuracy)

# predictions = np.argmax(activation2.output, axis=1)
# if len(y.shape) == 2:
#     y = np.argmax(y, axis=1)
# accuracy_code = np.mean(predictions==y)


# Print accuracy
# print('acc:', accuracy_code)


# ToDo{
# 	"1": input, weight, bias,
# 	"2": activation relu,
# 	"3": softmax,
#	"4-1": categorical cross Entropy,
#	"4-2": Loss,
#	"5": Accuracy,
# 	"6": Optimization,
# }

dense1 = Layer_Dense(2, 3)
dense2 = Layer_Dense(3, 3)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()
accuracy_function = Accuracy()




# 9397 . loss: 0.17279711 acc: 0.93
# 9878 . Loss: 0.268361 Accuracy: 0.083
# 8802 . loss: 0.17216808 acc: 0.93 Local result
