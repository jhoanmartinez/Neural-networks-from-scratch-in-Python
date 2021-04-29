import numpy as np

# Probabilities of 3 samples
softmax_outputs = np.array([[0.7, 0.2, 0.1],
                            [0.5, 0.1, 0.4],
                            [0.02, 0.9, 0.08]])
# Target (ground-truth) labels for 3 samples
class_targets = np.array([0, 1, 1])

#calcualr valores en el axis 1 (row)
predictions = np.argmax(softmax_outputs, axis = 1)

# si el target es hone hot encoded conevrted
if len(class_targets.shape) == 2:
	class_targets = np.argmax(class_targets, axis=1)

print(predictions)
print(class_targets)

accuracy = np.mean(class_targets == predictions)
print("accuracy:", accuracy)
