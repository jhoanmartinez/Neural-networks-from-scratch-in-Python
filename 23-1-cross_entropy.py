import math
import numpy as np

# Salida dede una capa de salida de red neuronal y^
softmax_output = [0.7, 0.1, 0.2]

# Ground truth yi
target_output = [1, 0, 0]

out = -(1*np.log(0.7) + 0*np.log(0.1) + 0*np.log(0.2))
out_arr = -np.sum((np.dot(target_output, np.log(softmax_output))))

print(out)
