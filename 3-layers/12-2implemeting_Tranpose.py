#cada weight contiene el numero de entradas
#el numero de weights debe ser igual a la
# misma cantidad de neurones o biases 

import numpy as np

inputs = np.array([     [1.0, 2.0, 3.0, 2.5],
                        [2.0, 5.0, -1.0, 2.0],
                        [-1.5, 2.7, 3.3, -0.8]])

weights = np.array([    [0.2, 0.8, -0.5, 1.0],
                        [0.5, -0.91, 0.26, -0.5],
                        [-0.26, -0.27, 0.17, 0.87],
                        [1,1,1,1],
                        [1,1,1,1]])

biases = np.array([2.0, 3.0, 0.5, 1, 1])

output = np.dot(inputs, np.array(weights).T)+biases

print("Inputs=>",inputs.shape)
print("Weights=>",weights.shape)
print("Biases=>",biases.shape)

#print(output)