import numpy as np

class Activation_Relu:

    # Forward pass
    def forward(self, inputs):
	    # Calculate output values from input
	    self.output = np.maximum(0, inputs)
