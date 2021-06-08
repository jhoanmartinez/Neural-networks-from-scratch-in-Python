import numpy as np

x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

#forward pass
xw0 = x[0]*w[0]
xw1 = x[1]*w[1]
xw2 = x[2]*w[2]
print("multi =",xw0, xw1, xw2, "bias=>",b)

#adding biase
z = xw0 + xw1 + xw2 + b
print("multi mas bias 1 =",z)

#ReLU activation
y = max(z, 0)
print("reulu 1 out =",y)

#Derivada de adelante por el valor de relu
dvalue = 1.0

#derivada de relu y chain rule
drelu_dz = dvalue * (1 if y > 0 else 0)
print("drelu dz =",drelu_dz)

# For the first partial derivative:
dsum_dxw0, dsum_dxw1, dsum_dxw2, dsum_db = 1, 1, 1, 1

drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db

print("dz * Xn+b =",drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

#derivawada de input X0*W0 * derivada entrante (drelu_dxwn)
dmul_dx0 = w[0]
dmul_dw0 = x[0]
dmul_dx1 = w[1]
dmul_dw1 = x[1]
dmul_dx2 = w[2]
dmul_dw2 = x[2]

drelu_x0 = (drelu_dz * dsum_dxw0) * dmul_dx0
drelu_w0 = (drelu_dz * dsum_dxw0) * dmul_dw0
drelu_x1 = (drelu_dz * dsum_dxw1) * dmul_dx1
drelu_w1 = (drelu_dz * dsum_dxw1) * dmul_dw1
drelu_x2 = (drelu_dz * dsum_dxw2)* dmul_dx2
drelu_w2 = (drelu_dz * dsum_dxw2)* dmul_dw2

print(drelu_x0, drelu_w0, drelu_x1, drelu_w1, drelu_x2, drelu_w2)


