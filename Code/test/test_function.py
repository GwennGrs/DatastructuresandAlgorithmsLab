import unittest
import numpy as np
from NeuralNetwork.function import sigmoid, sigmoid_derivatives, mse, gradient_descent, one_hot_encoder

class TestNeuralNetworkFunctions(unittest.TestCase):

    def test_sigmoid(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertAlmostEqual(sigmoid(1), 0.7310585786300049)
        self.assertAlmostEqual(sigmoid(-1), 0.2689414213699951)

    def test_sigmoid_derivatives(self):
        self.assertAlmostEqual(sigmoid_derivatives(0.5), 0.25)
        self.assertAlmostEqual(sigmoid_derivatives(0.7310585786300049), 0.19661193324148185)
        self.assertAlmostEqual(sigmoid_derivatives(0.2689414213699951), 0.19661193324148185)

    def test_mse(self):
        self.assertAlmostEqual(mse(self, np.array([1, 2, 3]), np.array([1, 2, 3])), 0.0)
        self.assertAlmostEqual(mse(self, np.array([1, 2, 3]), np.array([4, 5, 6])), 9.0)

    def test_gradient_descent(self):
        class MockModel:
            def __init__(self):
                self.weights = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
                self.derivatives = [np.array([0.1, 0.1]), np.array([0.1, 0.1])]

        model = MockModel()
        gradient_descent(model, learningRate=1)
        np.testing.assert_array_almost_equal(model.weights[0], np.array([1.1, 2.1]))
        np.testing.assert_array_almost_equal(model.weights[1], np.array([3.1, 4.1]))

    def test_one_hot_encoder(self):
        labels = np.array([0, 1, 2, 1, 0])
        one_hot_encoded = one_hot_encoder(labels)
        expected_output = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])
        np.testing.assert_array_equal(one_hot_encoded, expected_output)

if __name__ == '__main__':
    unittest.main()
