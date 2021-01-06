""" La exponenciacion ofrece multiples propositos, para calcular probabilidades
necesitamos valores siempre positivos, imagine la salida [3, 5, -7], aun
despues de normalizacion el ultimo valor siempre sera negativo dede que se
divide todos ellos por la suma de ellos

una probabiliad negativa no tiene sentido, Un valor exponencial de cualquier
numero sera siempre no negativo, devuelve 0 para un infinitos negativos """

# Values from the previous output when we described
# what a neural network is
"""layer_outputs = [4.8, 1.21, 2.385]"""
layer_outputs = [1, 1, 1]
# e - mathematical constant, we use E here to match a common coding
# style where constants are uppercased

""" y = e^x
	y = e^1  ó  y = e^layer_outputs[0] y continua con los otros valores...
"""

E = 2.71828182846 # you can also use math.e
# For each value in a vector, calculate the exponential value
exp_values = []

for k in layer_outputs:
	exp_values.append(E ** k) # [E^1, E^1, E^1] ** - power operator in Python

print('exponentiated values:')
print(exp_values)

# Now normalize values
norm_base = sum(exp_values)# We sum all values
print("\nsum of base norm: {}".format(norm_base))
norm_values = []

print("\n normalized values")
for i in exp_values:
	norm_values.append(i / norm_base)
	print(i, "/", norm_base, "=", i/norm_base)

print('\nNormalized exponentiated values:')
print(norm_values)
print('\nSum of normalized values:', sum(norm_values))
