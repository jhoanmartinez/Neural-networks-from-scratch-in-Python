import math

"""
Confidence level puede versi asi [0.22, 0.6, 0.18] o asi [0.32, 0.36, 0.32]
En ambos casos el argmax devolvera el segundo elemento como prediccion, pero
la confianza del modelo sobre estas predicciones es alta solo para ellos

La Categorical Cross-Entropy Loss saca mayor perdida a mayor desconfianza, y
menor perdida menor desconfianza

Cuando el nivel de confianza es de = 1 significa que el modelo esta 100% seguro
sobre la prediccion hecha, el valor de perdiad para esta muestra es = 0"""

print(math.log(1.))
print(math.log(0.95))
print(math.log(0.9))
print(math.log(0.8))
print('...')
print(math.log(0.2))
print(math.log(0.1))
print(math.log(0.05))
print(math.log(0.01))
