import numpy as np

# Permet de la non-linéarité
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Training data input
train = np.array([[0,0,1], [1,0,1], [1,1,0]])

# Block the "random"
np.random.seed(1)

# Training outputs
outputs = np.array([[0,1,0]]).T

# Training weight
fictive_weight = 2 * np.random.random((3,1)) -1

print("Training weight =", fictive_weight)
print ("- - - - - - - -")
print(sigmoid(np.dot(train, fictive_weight)))