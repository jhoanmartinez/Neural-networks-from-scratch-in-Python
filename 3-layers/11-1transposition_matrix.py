#la transpuesta es convertir las filas en columnas
#para poder tener las dimenisones requeridas para
#hacer el producto punto o multiplicar las matrices

import numpy as np

np.array([[1, 2, 3]])

a = [1,2,3]

print(np.array(a))

#expandir dimension con numpy
a = [1,2,3]
print(np.expand_dims(np.array(a), axis=0))