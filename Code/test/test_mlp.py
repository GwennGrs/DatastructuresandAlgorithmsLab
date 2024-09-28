from NeuralNetwork.multilayerPerceptron import MLP
import unittest

class TestMLP(unittest.TestCase):

    def setUp(self):
        # Initialize the MLP with some parameters
        self.mlp = MLP(input_size=3, hidden_size=5, output_size=2)

    def test_initialization(self):
        # Test if the MLP is initialized correctly
        self.assertEqual(self.mlp.input_size, 3)
        self.assertEqual(self.mlp.hidden_size, 5)
        self.assertEqual(self.mlp.output_size, 2)

    def test_forward_pass(self):
        # Test the forward pass with a sample input
        input_data = [0.5, -0.2, 0.1]
        output = self.mlp.forward(input_data)
        self.assertEqual(len(output), 2)  # Assuming output size is 2

    def test_backward_pass(self):
        # Test the backward pass with sample input and target
        input_data = [0.5, -0.2, 0.1]
        target = [1, 0]
        self.mlp.forward(input_data)
        self.mlp.backward(input_data, target)
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

if __name__ == '__main__':
    unittest.main()
