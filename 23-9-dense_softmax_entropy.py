import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
nnfs.init()

# X = np.array([
# 	[0.7,0.1],
# 	[0.1,0.5],
# 	[0.1,0.9],
# ])
# #
# y = np.array([0,1,1])

class Layer_Dense:

	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))

	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases
		return self.biases

class Activation_ReLU:

	def forward(self, inputs):
		self.output = np.maximum(0, inputs)
		return self.output

class Activation_Softmax:

	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities

class Loss:

	def calculate(self, output, y):
		sample_losses = self.forward(output, y)
		# print("output=",output[:3])
		# print("y=",y[:3])
		data_loss = np.mean(sample_losses)
		# print("Loss class - data losses", data_loss)
		return data_loss


class Loss_CategoricalCrossentropy(Loss):

	def forward(self, y_pred, y_true):
		samples = len(y_pred)
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

		if len(y_true.shape) == 1:

			#correct confidences is equals to 300 rows and 1 column
			correct_confidences = y_pred_clipped[
									range(samples),
									y_true
									]
			# print("correct_confidences 1=\n{} \n[\n{}, y_true{}\n]".format(y_pred_clipped[:4],range(samples), y_true) )
			# print("-------------\n",correct_confidences[:5])
			# print("shape confidences=",correct_confidences.shape)
			print("SOFTMAX CLIPED\n",y_pred_clipped,"\ny\n",y_true)
			print("total len of correct_confidences are {}\n ".format(len(correct_confidences)),correct_confidences)

		elif len(y_true.shape) == 2:
			correct_confidences = np.sum(
									y_pred_clipped*y_true,
									axis=1
									)
			print("correct_confidences 2=\n{} \n[\n{}, y_true{}\n]".format(y_pred_clipped[:4],range(samples), y_true) )

		negative_log_likelihoods = -np.log(correct_confidences)
		# print("Entropy class - negative_log_likelihoods",negative_log_likelihoods.shape)
		return negative_log_likelihoods

# Create dataset
X, y = spiral_data(samples=100, classes=3)
print("X shape=",X[:3].shape)
print("y shape=",y[:3].shape)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

# Create Softmax activation (to be used with Dense layer):
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

# Perform a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)

# Let's see output of the first few samples:
# print("activation 2 output\n",activation2.output[:3])

# Perform a forward pass through activation function
#takes the output of the activation function of the second dense layer as input
#and returns loss
loss = loss_function.calculate(activation2.output, y)

# Print loss value
print('\nloss\n', loss)



# correct_confidences 1=
# [[0.34098035 0.3242966  0.33472306]
#  [0.34098035 0.3242966  0.33472306]
#  [0.34098035 0.3242966  0.33472306]]
# [
# range(0, 3), y_true[0 0 0]
# ]

# a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#
# print(len(a))
