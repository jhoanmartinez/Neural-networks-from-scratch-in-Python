import numpy as np

class Loss:

	#calcula la peridad de datos y regilarizacion
	#valores de salida de verdad
	def calculate(self, output, y):

		#calcula la perdiad de las pmuestras
		sample_losses = self.forward(output, y)

		#calcula la media de la perdida
		data_loss = np.mean(sample_losses)

		#devuelve la media de la perdiad
		return data_loss
