from Code.NeuralNetwork.multilayerPerceptron import MLP
import unittest
import numpy as np

class TestMLP(unittest.TestCase):

    def setUp(self):
        # Initialize the MLP with some parameters
        self.mlp = MLP(num_inputs=3, hidden_layers=[5], num_outputs=2)

    def test_initialization(self):
        # Test if the MLP is initialized correctly
        self.assertEqual(self.mlp.num_inputs, 3)
        self.assertEqual(self.mlp.hidden_layers, [5])
        self.assertEqual(self.mlp.num_outputs, 2)

    def test_forward_pass(self):
        # Test the forward pass with a sample input
        input_data = [0.5, -0.2, 0.1]
        output = self.mlp.forward_propagate(input_data)
        self.assertEqual(len(output), 2)  # Assuming output size is 2

    def test_backward_pass(self):
        # Test the backward pass with sample input and target
        input_data = [0.5, -0.2, 0.1]
        target = [1, 0]
        self.mlp.forward_propagate(input_data)
        self.mlp.back_propagate(input_data, target)
        # Check if weights are updated (this is a simple check, you might want to check actual values)
        self.assertIsNotNone(self.mlp.weights_input_hidden)
        self.assertIsNotNone(self.mlp.weights_hidden_output)

    def test_training(self):
        # Test the training process
        training_data = [
            ([0.5, -0.2, 0.1], [1, 0]),
            ([0.3, 0.8, -0.5], [0, 1])
        ]
        self.mlp.train(training_data, epochs=10, learning_rate=0.01)
        # Check if the model has been trained (simple check)
        self.assertIsNotNone(self.mlp.weights_input_hidden)
        self.assertIsNotNone(self.mlp.weights_hidden_output)

         # Setup initial weights and derivatives for the test case
        self.weights = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        self.derivatives = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        
    def test_gradient_descent(self):
        # Expected weights after one gradient descent step with learningRate = 1
        expected_weights = [np.array([1.1, 2.2]), np.array([3.3, 4.4])]
        
        # Perform gradient descent
        self.gradient_descent(learningRate=1)
        
        # Check if the weights have been correctly updated
        for i in range(len(self.weights)):
            np.testing.assert_array_almost_equal(self.weights[i], expected_weights[i])

if __name__ == '__main__':
    unittest.main()
