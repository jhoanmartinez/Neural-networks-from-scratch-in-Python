input = [1, 2, 3]
weight = [0.2, 0.8, -0.5]
bias = 2
output = (input[0]*weight[0] + 
          input[1]*weight[1] + 
          input[2]*weight[2] + bias)
print(output)

input_2 = [output]
weight_2 = [0.3]
bias_2 = 3
output_2 = (input_2[0]*weight_2[0] + bias_2)
print(output_2)
