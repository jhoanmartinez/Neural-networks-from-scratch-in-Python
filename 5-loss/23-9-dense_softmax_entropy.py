import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
nnfs.init()

#================================
#			Layer Dense
#================================
class Layer_Dense:

	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.01*np.random.rand(n_inputs, n_neurons)
		self.biases = np.zeros( (1, n_neurons) )

	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases

#================================
#			ReLU
#================================
class Activation_ReLU:

	def forward(self, inputs):
		self.output = np.maximum(0, inputs)

#================================
#			Softmax
#================================
class Activation_Softmax:

	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities

#===========================================
#	Loss acces to Categorical_Cross_Entropy
#============================================
class Loss:

	def calculate(self, a, b):
		#a = softmax output, b = classes from dataset
		values = self.forward(a, b)
		data_loss = np.mean(values)
		return data_loss

#================================
#	Categorical cross Entropy
#================================
class Loss_CategoricalCrossentropy(Loss):

	def forward(self, y_pred, classes):
		samples = len(y_pred)
		y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
		# Probabilities for target values only if categorical labels
		if len(classes.shape) == 1:
			correct_confidences = y_pred_clipped[range(samples), classes]
		# Mask values - only for one-hot encoded labels
		elif len(classes.shape) == 2:
			correct_confidences = np.sum(y_pred_clipped * classes, axis=1)
		neg_log = -np.log(correct_confidences)
		return neg_log

#===========================================
#	Creando instancias y pasando datos
#============================================
# Create dataset
X = spiral_data(samples=100, classes=3)[0][:5]
y = spiral_data(samples=100, classes=3)[1][:5]
print("\nX => \n",X)

print("\ny => \n",y)

#==================================
# 		Proceso de la red
#=================================
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
dense2 = Layer_Dense(3, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()



# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Perform a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)


# Perform a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Perform a forward pass through activation function
# it takes the output of second dense layer here
# activation3.forward(dense2.output)

# Let's see output of the first few samples:
print(activation2.output[:5])

# Perform a forward pass through loss function
# it takes the output of second dense layer here and returns loss
loss = loss_function.calculate(activation2.output, y)

# Print loss value
print('loss:\n', loss)
print("\n activa2.out shape \n", activation2.output.shape)
print("\n activa2.out shape \n", y.shape)

# [[0.33333334 0.33333334 0.33333334]
#  [0.33333334 0.33333316 0.33333352]
#  [0.3333333  0.333333   0.33333367]
#  [0.3333333  0.33333305 0.3333336 ]
#  [0.33333334 0.3333329  0.33333376]]
# loss:
#  1.0986123

# [[0.0000000e+00 0.0000000e+00 0.0000000e+00]
#  [1.0421361e-06 5.1622737e-07 1.5931868e-06]
#  [2.1523676e-06 1.1326805e-06 3.2002497e-06]
#  [2.1542139e-06 1.3033393e-06 2.9727453e-06]
#  [3.1392894e-06 1.8559284e-06 4.3910072e-06]]
# loss:
#  1.0986123
