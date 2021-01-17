import numpy as np
import math

"""
	Volviendo a la funcion de perdida, necesitamos modificar la salida en 2
maneras. Primero se debe actualizar el proceso apra trabajar con batches con
distribuciones de salida Softmax; y Segundo, hacer el calculo negativo del
logaritmo dinamico para el indice objetivo
"""

""" Probabilidades por 3 muestras """
softmax_outputs = np.array([[0.7, 0.1, 0.2],
							[0.1, 0.5, 0.4],
							[0.02, 0.9, 0.08]])

"""
	necesitamos una manera dinamica de calcular la Categorical cross-Entropy
que como sabemos es un calculo negativo del logaritmo.

	Para determinar cual valor en la salida Softmax es para calcular el logaritmo
negativo, simplemente se debe saber el valor del objeivo

	En este ejemplo jay 3 clases, suponga que se esta tratando de clasificar
algo como un perro, gato y humano
Perro es clase 0 en indice 0,
Gato es clase 1 en indice 1,
Humano es clase 2 en indice 2,
Asumimos que el bacth de 3 muestras de entradas a la red neuronal al valor
objetivo de un perro, gato y gato.

Entonces el objetivo (como lista de indices de valores es) seria [0, 1, 1]
"""
index = {"perro":0, "gato":1, "humano":2}

softmax_outputs = np.array([[0.7, 0.1, 0.2], #dog
							[0.1, 0.5, 0.4], #cat
							[0.02, 0.9, 0.08]]) #cat

class_targets = np.array([0, 1, 1]) # dog, cat, cat

print("\nPrint from For")
for i, j in zip(class_targets, softmax_outputs):
	print(j[i])

print("\nPrint Slicing")
print(softmax_outputs[[0, 1, 2], class_targets])

print("\nPrint mixing variables")
print(softmax_outputs[range(len(softmax_outputs)), class_targets])

print("\nIndices de confianza")
neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
print(neg_log)
print("Exponentiandes Confidence 1.42857142294, 1.99999999888, 1.11111111594")

print("\nPromedio de perdida")
neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
average_loss = np.mean(neg_log)
print(average_loss)
