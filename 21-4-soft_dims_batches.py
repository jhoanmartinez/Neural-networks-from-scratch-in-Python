import numpy as np

#salidas de la capa anterior
layer_outputs = [4.8, 1.21, 2.385]

#exponenciar los valores
expo_values = np.exp(layer_outputs)

#Sumatoria de lso valores exponenciados
suma =  np.sum(expo_values)

#Normalizar los valores
norm_values = expo_values / suma

print("Exponencio = ",expo_values)
print("Normalizo = ",norm_values)
print("Suma de normalizados = ",np.sum(norm_values))

"""
Para entrenar en batches, necesitamos convertir esta funcionalidad
para aceptar salidas en capas de batches. Hacer esto es tan facil como:
"""

#obtener probabiliades no normalizada
#exp_values = np.exp(inputs)

#Normalizarlo para cada muestra(sample)
#probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

layer_outputs = np.array([	[4.8, 1.21, 2.385],
							[8.9, -1.81, 0.2],
							[1.41, 1.051, 0.026] ])

print('Sum without axis')
print(np.sum(layer_outputs))

print('This will be identical to the above since default is None:')
print(np.sum(layer_outputs, axis=None))

print('Another way to think of it w/ a matrix == axis 0: columns:')
print(np.sum(layer_outputs, axis=0))

#sumar filas en raw python
for i in layer_outputs:
	print(sum(i))

#sumar filas con numpy
print('So we can sum axis 1, but note the current shape:')
print(np.sum(layer_outputs, axis=1))

#mantener dimension de columna y no de fila
print('Sum axis 1, but keep the same dimensions as input:')
print(np.sum(layer_outputs, axis=1, keepdims=True))
