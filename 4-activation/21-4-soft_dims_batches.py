import numpy as np

#salidas de la capa anterior (la penultima, antes de la softmax)
layer_outputs = [4.8, 1.21, 2.385]

#exponenciar los valores
expo_values = np.exp(layer_outputs)

#Sumatoria de los valores exponenciados
suma =  np.sum(expo_values)

#Normalizar los valores
norm_values = expo_values/suma

# print("Exponencio = ",expo_values)
# print("Sumar = ", suma)
# print("Normalizo = exponencio/suma = ",norm_values)
# print("Suma de normalizados = ",np.sum(norm_values))

"""
Para entrenar en batches, necesitamos convertir esta funcionalidad
para aceptar salidas en capas de batches. Hacer esto es tan facil como:
"""

#obtener probabiliades no normalizada
#exp_values = np.exp(inputs)

#Normalizarlo para cada muestra(sample)
#probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

layer_outputs = np.array([	[1, 2, 3],
							[4, 5, 6]
						 ])
#
# print('Sum without axis')
# print(np.sum(layer_outputs))
#
# print('\nSum axis=None:')
# print(np.sum(layer_outputs, axis=None))
#
# print('\nSum axis=0:')
# print(np.sum(layer_outputs, axis=0))
#
# print('\nSum axis=1:')
# print(np.sum(layer_outputs, axis=1))
#
# print('\nDims input:')
# print(layer_outputs.shape)
#
# print('\nSum axis=0, Keepdims=True')
# a = np.sum(layer_outputs, axis=0, keepdims=True)
# print(a)
#
# print('\nDims output:')
# print(a.shape)

#sumar filas en raw python
# print("\nfor loop sum")
# for i in layer_outputs:
# 	print(sum(i))

#sumar filas con numpy
# print("\nSum axis=1 :")
# b = np.sum(layer_outputs, axis=1)
# print(b)
# print(b.shape)

#mantener dimension de columna y no de fila
# print("\nSum axis=1, keepdims=True, mantiene la dimension de la entrada")
# c = np.sum(layer_outputs, axis=1, keepdims=True)
# print(c)
# print(c.shape)

inputs = np.array([
					[5, 6],
					[7, 4],
					])

weights = np.array([
					 [1, 2, 3],
					 [4, 5, 6]
					 ])

biases = np.array([1, 1, 1])

output = np.dot(inputs, weights) + biases
print(output)
# [[0.12866601 0.32259424]
#  [0.34409058 0.73634647]]

class Activation_ReLU:

	def forward_relu(self, input):
		self.output = np.maximum(36, input)
		return self.output

class Softmax:

	def forward_softmax(self, input):
		exp_values = np.exp(input-np.max(input, axis=1, keepdims=True))
		print("\nExponentia values axis 1\n", exp_values)
		probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
		self.prob = probabilities
		return self.prob

a_r = Activation_ReLU()
print("\nRelu outpu\n",a_r.forward_relu(output))

a_s = Softmax()
print("\n Softmax\n",a_s.forward_softmax(a_r.output))
