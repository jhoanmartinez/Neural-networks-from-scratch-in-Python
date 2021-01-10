import numpy as np

#salidas de la capa anterior
layer_outputs = [4.8, 1.21, 2.385]

#exponenciar los valores
expo_values = np.exp(layer_outputs)

#Sumatoria de lso valores exponenciados
suma =  np.sum(expo_values)

#Normalizar los valores
norm_values = expo_values / suma

# print("Exponencio = ",expo_values)
# print("Normalizo = ",norm_values)
# print("Suma de normalizados = ",np.sum(norm_values))

"""
Para entrenar en batches, necesitamos convertir esta funcionalidad
para aceptar salidas en capas de batches. Hacer esto es tan facil como:
"""

#obtener probabiliades no normalizada
#exp_values = np.exp(inputs)

#Normalizarlo para cada muestra(sample)
#probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

layer_outputs = np.array([	[1, 1, 1],
						 ])

print('Sum without axis')
print(np.sum(layer_outputs))

print('\nSum axis=None:')
print(np.sum(layer_outputs, axis=None))

print('\nSum axis=0:')
print(np.sum(layer_outputs, axis=0))

print('\nDims input:')
print(layer_outputs.shape)

print('\nSum axis=0, Keepdims=True')
a = np.sum(layer_outputs, axis=0, keepdims=True)
print(a)

print('\nDims output:')
print(a.shape)

#sumar filas en raw python
print("\nfor loop sum")
for i in layer_outputs:
	print(sum(i))

#sumar filas con numpy
print("\nSum axis=1 :")
b = np.sum(layer_outputs, axis=1)
print(b)
print(b.shape)

#mantener dimension de columna y no de fila
print("\nSum axis=1, keepdims=True, mantiene la dimension de la entrada")
c = np.sum(layer_outputs, axis=1, keepdims=True)
print(c)
print(c.shape)
