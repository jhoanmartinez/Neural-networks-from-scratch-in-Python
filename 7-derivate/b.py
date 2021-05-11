import numpy as np
import matplotlib.pyplot as plt

def f(d):
    return 2*x**2

x = np.array(range(-2,5,1))
y = f(x)

print(y)

plt.plot(x, y)
plt.grid(True)
plt.show()

# X = [-2 -1  0  1  2  3  4]
# f(x) = [ 8  2  0  2  8 18 32]
