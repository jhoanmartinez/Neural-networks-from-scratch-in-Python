import numpy as np

#inputs 4x3
#Transpose input 4x3
inputs = np.array([ [4,9],
                    [3,3],
                    [3,5],
                    [3,7] ])

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def visual(self):
        print("\nWeights")
        print(self.weights.shape)
        print("\nBiases")
        print(self.biases.shape)
        print("\nOutput")
        print(self.output.shape)


dense_1 = Layer_Dense(2,3)
dense_1.forward(inputs)
print("X inputs")
print(inputs.shape)
dense_1.visual()

#Entradas son las columnas de cada array
#Layer_Dense(3, 4)
# X inputs
# (4, 3)
# [[4 5 9]
#  [3 8 3]]

# Weights
# (3, 6)

# Biases
# (1, 6)

# Output
# (4, 6)

#Layer_Dense(2, 3)
# X inputs
# (4, 2)
# [[4 9]
#  [3 3]]

# Weights
# (2, 3)

# Biases
# (1, 3)

# Output
# (4, 3)
