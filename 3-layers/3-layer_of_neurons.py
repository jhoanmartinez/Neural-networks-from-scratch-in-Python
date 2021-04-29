#cada entrada tiene un peso 
#cuantos pesos entran?
#cada neuron tiene un bias
#cuatro entradas y cada una con 4 pesos
#suman 1 neuron y sale un array de 3 valores
inputs = [1, 2, 3, 2.5]

weights_1 = [0.2, 0.8, -0.5, 1]
weights_2 = [0.5, -0.91, 0.26, -0.5]
weights_3 = [-0.26, -0.27, 0.17, 0.87]

biases = [2,3,0.5]

output = [
    #neuron 1
    inputs[0]*weights_1[0]+
    inputs[1]*weights_1[1]+
    inputs[2]*weights_1[2]+
    inputs[3]*weights_1[3]+biases[0],

    #neuron 2
    inputs[0]*weights_2[0]+
    inputs[1]*weights_2[1]+
    inputs[2]*weights_2[2]+
    inputs[3]*weights_2[3]+biases[1],

    #neuron 3
    inputs[0]*weights_3[0]+
    inputs[1]*weights_3[1]+
    inputs[2]*weights_3[2]+
    inputs[3]*weights_3[3]+biases[2]

]

print(output)

