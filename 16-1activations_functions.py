""" Sin la funcion de activacion todo seria lineal
    esto permite a las capas ocultas mapear funciones
    no lineales """

#la funcion de activacion se le aplica a la salida
# de la neurona o capa de nuerones, el cual modifica la salida

"""hay dos tipos de "activations functions", las que estan a la
salida del neuron o capa de neuronas y las que estan a la salida
de toda la arquitectura"""

#usualmente es la misma para todos pero no siempre es asi

""" una funcion de activacion va a disparar uno o creo de acuerdo
    a ka ibnfomacion que tiene en la entrada"""

#step activation function (Escalon) o (Escalon simetrico)
""" si input * weight + bias (la salida del neuron) es > 0 el 
    neuron se activa y salida = 1 si la salida del neuron 
    es <= 0 el neuron no se activa y salida = 0
    
    y = | 1 x > 0   
        | 0 x < 0 
    """

#linear activation function (Lineal) 
""" la salida es igual a la entrada x = y la salida es igual a 
    lo que viene en la entrada
    
    y = x 
    """

#sigmoid activation function (Logaritmica sigmoide) o (tansigmoide)
""" El problema con una función escalonada es
que el optimizador tiene menos claro cuáles son estos impactos porque 
hay muy poca información recopilada de esta función. Está activado (1) o 
desactivado (0). Es difícil decir qué tan "cerca" estuvo esta 
función de paso de activar o desactivar. Tal vez estuvo muy cerca, o 
tal vez estuvo muy lejos. En términos del valor de salida final de 
la red, no importa si estuvo cerca de producir algo más. Por lo tanto, 
cuando llega el momento de optimizar las ponderaciones y los sesgos, 
es más fácil para el optimizador si tenemos funciones de activación 
más granulares e informativas. 
    devuelve 0 < 0
    devuelve 0 < 0.5
    devuelve 1 > 0.5
para neuronas muertas es mejor tener mas funciones como la sigmoide

    y = 1 / 1 + e^x

Rectified Linear Units activation function (or ReLU)
es mas simple que la sigmoiden, por poco es literalmente 
igual y = x, empzando en cero desde el lado negativo, es la mas usada
por su velocidad y eficiencia, la sigmpide no es muy complicada pero la
ReLU es mas facil para hacer su computacion, es casi una funcion lineal 
pero se mantiene no lineal, debido a esa inclinacion despues de cero, esa
propiedad es muy efectiva

    y = | x x >  0
        | 0 x <= 0
    """


#rectified activation function (Rectificada)


import numpy as np

sensor_1 = [1,1,1]
sensor_2 =  [10,10,10]

inputs = np.array([sensor_1, sensor_2])

class Layer_Dense:
    
    def __init__(self, n_neurons, n_weights):
        self.weights = np.ramdon.randn(n_neurons, n_weights)
        self.biases = np.zeros((n_neurons))
    
    def forward(self, data):
        self.output = np.dot(data, self.weights) + self.biases
        return self.output