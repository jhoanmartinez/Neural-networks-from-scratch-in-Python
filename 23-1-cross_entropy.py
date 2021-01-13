import math

# Salida dede una capa de salida de red neuronal
softmax_output = [0.7, 0.1, 0.2]

# Ground truth
target_output = [1, 0, 0]

loss = -(
		target_output[0]*math.log(softmax_output[0])+
		target_output[1]*math.log(softmax_output[1])+
		target_output[2]*math.log(softmax_output[2])
)

loss_2 = -(target_output[0] * math.log(softmax_output[0]))

print(loss)
print(loss_2)
