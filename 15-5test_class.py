#cada weight contiene el numero de entradas
#el numero de entradas es el numero de columnas del input
#el numero de weights debe ser igual a la
# misma cantidad de neurones o biases


import numpy as np

#input 4x3
sensor_1 =  [1,1,1,1,1]
sensor_2 =  [1,1,1,1,1]
sensor_3 =  [1,1,1,1,1]
sensor_4 =  [1,1,1,1,1]

#entradas de 4x2
inputs = np.array([
    sensor_1,
    sensor_2,
    sensor_3,
    sensor_4
])



class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        # self.weights = np.random.randn(n_inputs, n_neurons)
        # self.biases = np.zeros((n_neurons))
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, data):
        self.output = np.dot(data, self.weights) + self.biases
        return self.output

    def visual(self):
        print("Weights=>shape",self.weights.shape)
        print(self.weights)
        print("Biases=>shape",self.biases.shape)
        print(self.biases)

print("inputs=>shape",inputs.shape)
print(inputs)

hidden_1 = Layer_Dense(5, 7)
hidden_1.visual()
print("Forward output")
print(hidden_1.forward(inputs))
