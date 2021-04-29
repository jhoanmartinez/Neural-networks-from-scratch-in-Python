#no enetndi esto y utilice el array de abahjo llamado sensores
import nnfs
from nnfs.datasets import spiral_data
import numpy as np
import matplotlib.pyplot as plt

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()

#para los datos se utilizo ese ejemplo
import numpy as np

#input 4x2
sensor_1 =  [1,1]
sensor_2 =  [1,1]
sensor_3 =  [1,1]
sensor_4 =  [1,1]

#entradas de 4x2
inputs = np.array([
    sensor_1,
    sensor_2,
    sensor_3,
    sensor_4
])

#https://cs231n.github.io/neural-networks-case-study/
