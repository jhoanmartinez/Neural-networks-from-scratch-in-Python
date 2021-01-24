import numpy as np

softmax_outputs = np.array([
								[0.7, 0.1, 0.2],
								[0.1, 0.5, 0.4],
								[0.02, 0.9, 0.08]
							])

class_targets = np.array([
							[1, 0, 0],
							[0, 1, 0],
							[0, 1, 0]
						])

#Probabilidades para target values
#solamente si Categorical labels
if len(class_targets.shape) == 1:
	correct_confidences = softmax_outputs[range
										(len((softmax_outputs)),
										class_targets]

#marcar los valores solo para one-hot encoded
elif len(class_targets.shape) == 2:
	correct_confidences = np.sum(softmax_outputs*class_targets, axis=1)

#perdida
neg_log = -np.log(correct_confidences)

#promedio de peridad
average_loss = np.mean(neg_log)
print(average_loss)
