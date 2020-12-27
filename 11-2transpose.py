import numpy as np
 
a = [1,2,3]
b = [2,3,4]
print(np.dot(a,b))

a = np.array([a])
b = np.array([b]).T
print(np.dot(a,b))