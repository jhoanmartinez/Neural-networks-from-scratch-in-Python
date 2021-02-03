import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
							[0.1, 0.5, 0.4],
							[0.02, 0.9, 0.08]])

class_targets = np.array([	[1, 0, 0],
							[0, 1, 0],
							[0, 1, 0] ])


y = np.array([2,4,3])
print("Y shape = ",y.shape)

print(len(class_targets.shape))

#Common loss class
class Loss:

	#calcula la regulariazion de la perdida
	#de los datos dando la salida del modelo
	#y los valores true o one-hot
	def calculate(self, output, y):

		#clacular la perdiad de las muestras
		sample_losses = self.forward(output, y)
		print("output=",output)
		print("y=",y)

		#calcular la media de la perdida
		data_loss = np.mean(sample_losses)

		#devolver la perdida
		return data_loss

#Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

	#forward pass
	def forward(self, y_pred, y_true):

		#numero de muestras por batch
		samples = len(y_pred)
		print("BATCH SAMPLES =", samples)

		#clip data para rpevenir division por cero
		#clip minimo y maximo (ambos lados)
		y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

		# Probabilities for target values -
		# only if categorical labels
		if len(y_true.shape) == 1:
			print("y_true if 1=",len(y_true.shape))
			correct_confidences = y_pred_clipped[ range(samples), y_true]
			print("correct_confidences 1=\n{} \n[\n{}, y_true{}\n]".format(y_pred_clipped[:4],range(samples), y_true) )

		# Mask values - only for one-hot encoded labels
		elif len(y_true.shape) == 2:
			print("y_true if 2=",len(y_true.shape))
			correct_confidences = np.sum( y_pred_clipped*y_true, axis=1)
			print("correct_confidences 2=\n{} \n[\n{}, y_true{}\n]".format(y_pred_clipped[:4],range(samples), y_true) )

			print("--------",correct_confidences)

		# Losses
		negative_log_likelihoods = -np.log(correct_confidences)
		return negative_log_likelihoods


loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(softmax_outputs, class_targets)
# print(loss)
