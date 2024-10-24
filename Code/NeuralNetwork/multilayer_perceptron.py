"""
This module implements a Multilayer Perceptron (MLP) neural network from scratch using numpy. 
It includes methods for forward propagation, backpropagation, and training the network.
Classes:
    MLP: A class representing a Multilayer Perceptron neural network.
Functions:
    sigmoid: Sigmoid activation function.
    sigmoid_derivatives: Derivative of the sigmoid function.
    mse: Mean Squared Error loss function.

"""

import numpy as np
from Code.NeuralNetwork.function import sigmoid, sigmoid_derivatives, mse

class MLP():
    """A Multilayer Perceptron class.
    """

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs

        Arguments:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        """
        # Set the random seed for reproducibility

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # to simplify the representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random connection weights for the layers,
        # segments the "layer" array into the necessary matrices.
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        # save derivatives per layer
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        for layer in enumerate(layers):
            a = np.zeros(layer)
            activations.append(a)
        self.activations = activations


    def forward_propagate(self, inputs):
        """Computes forward propagation of the network based on input signals.

        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """

        # the input layer activation is just the input itself
        activations = inputs

        # save the activations(here the inputs) for backpropagation
        self.activations[0] = activations

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = sigmoid(net_inputs)

            # save the activations for backpropagation
            self.activations[i + 1] = activations

        # return output layer activation
        return activations


    def back_propagate(self, error):
        """Backpropagates an error signal.
        Args:
            error (ndarray): The error to backpropagate.
        Returns:
            error (ndarray): The final error of the input
        """

        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):

            # get activation for previous layer
            activations = self.activations[i+1]

            # apply sigmoid derivative function
            delta = error * sigmoid_derivatives(activations)

            # reshape delta (-1 guess automatically the size)
            delta_tmp = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(current_activations.shape[0],-1)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_tmp)

            # backpropagate the next error
            error = np.dot(delta, self.weights[i].T)


    def train(self, inputs, targets, epochs, learning_rate):
        """Trains model running forward prop and backpropagation
        Args:
            inputs (ndarray): X
            targets (ndarray): Y
            epochs (int): Num. epochs we want to train the network for
            learning_rate (float): Step to apply to gradient descent
        """
        # now enter the training loop
        for i in range(epochs):
            sum_errors = 0

            # iterate through all the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # activate the network!
                output = self.forward_propagate(input)


                error = target - output

                self.back_propagate(error)

                # now perform gradient descent on the derivatives
                # (this will update the weights
                self.gradient_descent(learning_rate)

                # keep track of the MSE for reporting later
                sum_errors += mse(target, output)

            # Print the progress
            self.print_progress(i + 1, epochs)

        print("\nTraining complete!")
        print("=====")

    def print_progress(self, current_epoch, total_epochs):
        """Prints the progress of the training."""
        percent_complete = (current_epoch / total_epochs) * 100
        bar_length = 50  # Length of the progress bar
        block = int(round(bar_length * current_epoch / total_epochs))
        # Create the progress bar string
        progress_bar = f"[{'#' * block}{'-' * (bar_length - block)}] {percent_complete:.2f}%"

        # Print the progress bar in the same line
        print(f"\r{progress_bar}", end="")  # `end=""` to stay on the same line

    def gradient_descent(self, learning_rate=1):
        """Learns by descending the gradient
        Args:
            learningRate (float): How fast to learn.
        """
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            self.weights[i] += self.derivatives[i] * learning_rate
