import numpy as np
import matplotlib.pyplot as plt

def f(d):
    return 2*x

x = np.array(range(0,5,1))
y = f(x)

print(x)
print(y)

print((y[1]-y[0]) / (x[1]-x[0]))
print((y[1]-y[0]) / (x[1]-x[0]))

y1 = f(x)  # result at the derivation point

plt.plot(x, y)
plt.grid(True)
plt.show()

# X = [-2 -1  0  1  2  3  4]
# f(x) = [ 8  2  0  2  8 18 32]
