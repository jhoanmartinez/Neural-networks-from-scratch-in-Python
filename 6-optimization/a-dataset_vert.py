import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data

nnfs.init()
X = vertical_data(samples=100, classes=3)[0]
y = vertical_data(samples=100, classes=3)[1]

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()
