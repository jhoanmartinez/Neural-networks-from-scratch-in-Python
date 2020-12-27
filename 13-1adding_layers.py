import numpy as np
#la red neuronal se convierte en deep cuando tiene mas de 2 hiddens layers
#o capas ocultas

#entradas primera capa
inputs = [  [1, 2, 3, 2.5],
            [2., 5., -1., 2],
            [-1.5, 2.7, 3.3, -0.8]]

weights = [ [0.2, 0.8, -0.5, 1],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer_1_outputs = [np.dot(inputs, np.array(weights).T)+biases]

print("Layer 1 output")
print(layer_1_outputs)

#entradas segunda capa
inputs_2 = layer_1_outputs

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer_2_output = [np.dot(inputs_2, np.array(weights2).T)+biases2]
print("Layer 2 output")
print(layer_2_output)