import numpy as np
from function import sigmoid_derivatives, sigmoid

# Training data input
train = np.array([[0,0,1], [1,0,1], [1,1,0]])

# Block the "random"
np.random.seed(1)

# Training outputs
wanted_outputs = np.array([[0,1,0]]).T

# # Training weight

fictive_weight = 2 * np.random.random((3,1)) -1
print("Old training weight =", fictive_weight)
print ("- - - - - - - -")
print("Outputs avant train", sigmoid(np.dot(train, fictive_weight)))
print ("- - - - - - - -")


for i in range(100000):
    input = train
    outputs = sigmoid(np.dot(input, fictive_weight))

    error = wanted_outputs - outputs

    adjustements =  error * sigmoid_derivatives(outputs)

    fictive_weight += np.dot(input.T, adjustements)

print("New training weight =", fictive_weight)
print ("- - - - - - - -")
print("Final outputs = ", outputs)