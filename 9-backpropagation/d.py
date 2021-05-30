# Calculating the gradients with respect to weights is very similar, 
# but, in this case, we’re going to be using gradients to update the 
# weights, so we need to match the shape of weights, not inputs. Since 
# the derivative with respect to the weights equals inputs, weights 
# are transposed, so we need to transpose inputs to receive the derivative 
# of the neuron with respect to weights. Then we use these transposed 
# inputs as the first parameter to the dot product — the dot product is 
# going to multiply rows by inputs, where each row, as it is transposed, 
# contains data for a given input for all of the samples, by the columns 
# of dvalues. These columns are related to the outputs of singular neurons 
# for all of the samples, so the result will contain an array with the 
# shape of the weights, containing the gradients with respect to the inputs, 
# multiplied with the incoming gradient for all of the samples in the batch:

import numpy as np

# Passed-in gradient from the next layer
# for the purpose of this example we're going to use
# an array of an incremental gradient values
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

# We have 3 sets of inputs - samples
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])

# sum inputs for given weight
# and multiply by the passed-in gradient for this neuron
dweights = np.dot(inputs.T, dvalues)

print(dweights)
