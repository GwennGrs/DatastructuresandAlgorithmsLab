import unittest
import numpy as np
from Code.NeuralNetwork.multilayer_perceptron import MLP

class TestMLP(unittest.TestCase):
        
        def setUp(self):
            # Initialize the MLP with some parameters
            self.mlp = MLP(num_inputs=3, hidden_layers=[5, 3], num_outputs=2)
            # Set predefined weights to ensure consistent testing
            self.mlp.weights = [
                np.array([[0.2, 0.4, 0.6, 0.8, 0.5],
                          [0.1, 0.3, 0.5, 0.7, 0.2],
                          [0.9, 0.2, 0.1, 0.4, 0.6]]),  # Weights for layer 1
                np.array([[0.3, 0.7, 0.2],
                          [0.6, 0.2, 0.5],
                          [0.8, 0.4, 0.9],
                          [0.5, 0.9, 0.3],
                          [0.2, 0.3, 0.6]]),  # Weights for layer 2
                np.array([[0.3, 0.7],
                          [0.6, 0.2],
                          [0.8, 0.4]])  # Weights for output layer
            ]

        
        def test_initialization(self):
            # Test if the MLP is initialized correctly
            self.assertEqual(self.mlp.num_inputs, 3)
            self.assertEqual(self.mlp.hidden_layers, [5, 3])
            self.assertEqual(self.mlp.num_outputs, 2)

        def test_forward_pass(self):
            # Test the forward pass with a sample input
            input_data = [0.5, -0.2, 0.1]
            output = self.mlp.forward_propagate(input_data)
            self.assertEqual(len(output), 2)  # Assuming output size is 2
            self.assertAlmostEqual(output[0], 0.79549026, places=5)  # Testing output value

        def test_back_propagate(self):
            # Simulate a forward pass to store activations
            input_data = np.array([0.5, 0.1, 0.9])
            self.mlp.forward_propagate(input_data)

            # Define a sample error signal (difference between target and output)
            error = np.array([0.3, -0.1])

            # Perform backpropagation
            self.mlp.back_propagate(error)

            # Check the shape of the derivatives to match weights structure
            self.assertEqual(self.mlp.derivatives[0].shape, self.mlp.weights[0].shape)
            self.assertEqual(self.mlp.derivatives[1].shape, self.mlp.weights[1].shape)
            self.assertEqual(self.mlp.derivatives[2].shape, self.mlp.weights[2].shape)

            # Check if the derivatives are different from zero (initialized as zeros)
            for d in self.mlp.derivatives:
                self.assertFalse(np.array_equal(d, np.zeros(d.shape)))

        def test_train(self):
            # Sample input data (X) and target data (Y)
            inputs = np.array([
                [0.5, -0.2, 0.1],
                [0.9, 0.1, 0.4],
                [0.3, 0.8, -0.5]
            ])
            
            targets = np.array([
                [0.1, 0.9],
                [0.8, 0.2],
                [0.4, 0.6]
            ])

            # Store initial weights
            initial_weights = [w.copy() for w in self.mlp.weights]

            # Train the MLP
            self.mlp.train(inputs, targets, epochs=10, learning_rate=0.01)

            # Check if the weights have been updated (i.e., not equal to initial weights)
            for i in range(len(self.mlp.weights)):
                self.assertFalse(np.array_equal(self.mlp.weights[i], initial_weights[i]))

if __name__ == '__main__':
        unittest.main()