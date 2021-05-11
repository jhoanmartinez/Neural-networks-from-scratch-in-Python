import numpy as np
import matplotlib.pyplot as plt

def f(param):
    return 2*param**2

x = np.arange(0, 5, 0.001)
y = f(x)

print(x)
print(y)

plt.plot(x, y, a)
plt.grid(True)
plt.show()
