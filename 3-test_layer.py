inputs = [2,6,3]

weights_1 = [4,7,6]
weights_2 = [1,8,5]

biases = [2,3]

outputs = [
    #neuron 1
    inputs[0]*weights_1[0]+
    inputs[1]*weights_1[1]+
    inputs[2]*weights_1[2]+ biases[0],

    #neuron 2
    inputs[0]*weights_2[0]+
    inputs[1]*weights_2[1]+
    inputs[2]*weights_2[2]+ biases[1]
]

print(outputs)