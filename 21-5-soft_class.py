import numpy as np

"""Lo que la funcion softmax de vuelve son las probabilidades de cada clase"""

class Activation_Softmax:

	#forward pass
	def forward(self, inputs):

		# Get unnormalized probabilities
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

		# Normalize them for each sample
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities

"""
Entre el valor mas alto para el e, va a existir overflow por ser un valor muy
grande a la salida, por eso se debe normalizar para evitar estos errores
"""

# print("1 ==>",np.exp(1))
# print("10 ==>",np.exp(10))
# print("100 ==>",np.exp(100))
# print("1000 ==> overflow error")
#
softmax = Activation_Softmax()
softmax.forward([[1,2,3]])
print("softmax output 1 ==>",softmax.output)
#
softmax.forward([[-2, -1, 0]]) # subtracted 3 - max from the list
print("softmax output 2 ==>",softmax.output)
#
softmax.forward([[0.5, 1, 1.5]])
print("softmax output 3 ==>",softmax.output)
