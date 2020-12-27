#cuantos pesos entran?
#cada entrada tiene un peso
#cada neuron tiene un bias

inputs = [3, 4]

weights_1 = [2,3]
weights_2 = [7,5]
weights_3 = [2,4]
weights_4 = [4,1]

biases = [3,6,4,7]

outputs = [
    #neuron 1
    inputs[0]*weights_1[0]+
    inputs[1]*weights_1[1]+biases[0],

    #neuron 2
    inputs[0]*weights_2[0]+
    inputs[1]*weights_2[1]+biases[1],

    #neuron 3
    inputs[0]*weights_3[0]+
    inputs[1]*weights_3[1]+biases[2],

    #neuron 3
    inputs[0]*weights_4[0]+
    inputs[1]*weights_4[1]+biases[3]

]

print(outputs)