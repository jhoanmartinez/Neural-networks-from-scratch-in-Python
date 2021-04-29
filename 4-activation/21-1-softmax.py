"""
	función softmax, o función exponencial normalizada
es una generalización de la Función logística. Se emplea para "comprimir" un
vector K-dimensional, de valores reales arbitrarios en un vector K-dimensional,
 de valores reales en el rango [0, 1]

En teoría de la probabilidad, la salida de la función softmax puede ser
utilizada para representar una distribución categórica– la distribución de
probabilidad sobre diferentes posibles salidas.


	El primer paso es exponenciar las salidas con la constante de euler el cual es
2.71828182846 y se refiere al crecimiento exponencial, exponenciando es tomando
esta tomando esta constante a el poder de el parametro dado

y = e^x

El numerador y denominador con tiene e elevado a la z, donde:
z son los indices dados, siginifican una salida singular media
i siginifica la muestra actual
j siginifica la actual salida en esta muestra

softmax formula

el numerador exponencia el valor de la salida actual y el denominador toma la
suma de todos los las salidas para una muestra dada

"""
# Values from the previous output when we described
# what a neural network is
#layer_outputs = [4.8, 1.21, 2.385]
layer_outputs = [1, 1, 1]

# e - mathematical constant, we use E here to match a common coding
# style where constants are uppercased
e = 2.71828182846 # you can also use math.e

"""Para cada valor en un vector calcular el valor exponencial"""
exp_values = []
for output in layer_outputs:
	exp_values.append(e ** output) # ** - power operator in Python
print('exponentiated values:')
print(exp_values)
