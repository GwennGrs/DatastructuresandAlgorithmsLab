import unittest
import numpy as np
from Code.NeuralNetwork.function import sigmoid, sigmoid_derivatives, mse, one_hot_encoder

class TestNeuralNetworkFunctions(unittest.TestCase):

    def test_sigmoid(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertAlmostEqual(sigmoid(1), 0.7310585786300049)
        self.assertAlmostEqual(sigmoid(-1), 0.2689414213699951)

    def test_sigmoid_derivatives(self):
        self.assertAlmostEqual(sigmoid_derivatives(0.5), 0.25)
        self.assertAlmostEqual(sigmoid_derivatives(0.7310585786300049), 0.19661193324148185)
        self.assertAlmostEqual(sigmoid_derivatives(0.2689414213699951), 0.19661193324148185)


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
        
    def test_mse_perfect_match(self):
        # Cas où les cibles et les sorties sont exactement les mêmes (MSE doit être 0).
        target = np.array([1, 1, 1, 1])
        output = np.array([1, 1, 1, 1])
        self.assertAlmostEqual(mse(target, output), 0.0, places=6)

    def test_mse_with_differences(self):
        # Cas général où il y a une différence entre les cibles et les sorties.
        target = np.array([1, 0, 0, 1])
        output = np.array([0.9, 0.1, 0.2, 0.8])
        self.assertAlmostEqual(mse(target, output), 0.025, places=6)

    def test_mse_extreme_difference(self):
        # Cas extrême où les cibles sont opposées aux sorties (1 contre 0 et inversement).
        target = np.array([1, 1, 1, 1])
        output = np.array([0, 0, 0, 0])
        self.assertAlmostEqual(mse(target, output), 1.0, places=6)

    def test_mse_mixed_values(self):
        # Cas général avec des valeurs positives, négatives et des petites différences.
        target = np.array([1, -2, 3, -4])
        output = np.array([1.1, -1.9, 2.9, -4.1])
        self.assertAlmostEqual(mse(target, output), 0.01, places=6)

    if __name__ == '__main__':
        unittest.main()
if __name__ == '__main__':
    unittest.main()
