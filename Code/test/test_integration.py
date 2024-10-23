from Code.NeuralNetwork.multilayerPerceptron import MLP
from Code.input.titanic.prep_data import prep_traindata, test_user_input, testdata
import unittest
import pandas as pd 

class TestNeuralNetworkFunctions(unittest.TestCase):

    # Preparing data for MLP model
    X_train, y_train = prep_traindata()

    # The number of outputs must be 3 (one-hot encoding for classes 1, 2, 3)
    mlp = MLP(num_inputs=X_train.shape[1], hidden_layers=[3, 3], num_outputs=2)  # 2 exits for classes 0 (not survived) and 1 (survived)
    # Training the MLP model
    mlp.train(X_train, y_train, epochs=400, learning_rate=0.1)    

    ## Testing on input data
    
    # First one of the test_fusion dataset supposed to be a non-survivor
    user_input = {
            'Pclass': 3,
            'Sex': 0 ,  
            'Age': 34.5,
            'SibSp': 0,
            'Parch': 0,
            'Fare': 7.8292,
            'Embarked': {'C': 0, 'Q': 1, 'S': 2}["Q"]  # Convert 'C', 'Q', 'S' to 0, 1, 2
        }

    # Assert the prediction based on the input data

    def test_user_input(self):
        self.assertEqual(test_user_input(self.mlp, self.user_input), 0)
    
    # Test the precision of the model on the test data > 80%

    def test_model_precision(self):
        self.assertGreater(testdata(self.mlp), 0.8)
        
if __name__ == '__main__':
    unittest.main()

    

