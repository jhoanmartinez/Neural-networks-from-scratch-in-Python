import numpy as np

# Valors previos cuando describimos lo que una red
#neuronal es
# layer_outputs = [4.8, 1.21, 2.385]
#
# #para cada valor en el vector alcular el valor exponencial
# exp_values = np.exp(layer_outputs)
# print("valores exponenciales:")
# print(exp_values)
#
# #normalizacion de valores
# norm_values = exp_values / np.sum(exp_values)
# print("\nvalores exponenciales normalizados:")
# print(norm_values)
#
# #suma de los valores normalizados
# print("\nsuma de los avlores normalizados")
# print(np.sum(norm_values))

""" El valor normalizado es el porcentaje de porbabilidad """

#salidas de la capa anterior
layer_outputs = [-1, -2, 0]

#exponenciar los valores
expo_values = np.exp(layer_outputs)

#Sumatoria de lso valores exponenciados
suma =  np.sum(expo_values)

#Normalizar los valores
norm_values = expo_values / suma

print("Exponencio = ",expo_values)
print("Normalizo(probablidad) = ",norm_values)
print("Suma de normalizados = ",np.sum(norm_values))

#Normalizo(probablidad) =  [0.09003057 0.24472847 0.66524096]
#Normalizo(probablidad) =  [0.24472847 0.09003057 0.66524096]
