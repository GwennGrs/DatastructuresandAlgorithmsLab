import numpy as np

## For the MLP model

# For nonlinearity
def sigmoid(x):
    """
    Compute the sigmoid of x.
    Parameters:
        x (numpy.ndarray): Input array.
    Returns:
        numpy.ndarray: The sigmoid of the input array.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivatives(x):
    """
    Compute the derivative of the sigmoid function.
    Parameters:
        x (numpy.ndarray): Input array.
    Returns:
        numpy.ndarray: The derivative of the sigmoid function.
    """
    return x*(1-x)

def mse(target, output):
    """
    Compute the Mean Squared Error (MSE) between the target and output.
    Parameters:
        target (numpy.ndarray): The target values.
        output (numpy.ndarray): The predicted values.
    Returns:
        float: The mean squared error.
    """
    return np.average((target - output) ** 2)

# For data preprocessing

# OneHotEncoder
def one_hot_encoder(labels):
    """
    Encode labels into a one-hot representation.
    Parameters:
        labels (numpy.ndarray): Array of labels to be encoded.
    Returns:
        numpy.ndarray: One-hot encoded representation of the input labels.
    """
    unique_labels = np.unique(labels)
    one_hot_encoded = np.zeros((len(labels), len(unique_labels)))
    
    for i, label in enumerate(labels):
        index = np.where(unique_labels == label)[0][0]
        one_hot_encoded[i, index] = 1
    
    return one_hot_encoded
