import numpy as np

#log 0
print(-np.log(0))

print(np.e**(-np.inf))

#error de valor infinito
print(np.mean([1,2,3,-np.log(0)]))

#agregar valor para qu eno sea infinito
print(-np.log(1e-7))

#agregando un valor para que no impacte el resultado
#proble en al caso donde la confianza es 1 no alcanza
print(-np.log(1+1e-7))

#convertir a uno un posilbe cero como label correcto
print(-np.log(1-1e-7))


y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
print(y_pred_clipped)
