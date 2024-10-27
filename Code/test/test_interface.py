import unittest
from unittest.mock import MagicMock, patch
from Code.interface_package.interface import tester_modele, afficher_donnees

class TestTesterModele(unittest.TestCase):
    def setUp(self):
        # I set up the test by creating mock objects for the MLP model, test data, and label
        self.mlp = MagicMock()  
        # I simule the testdata return value 
        self.testdata = MagicMock(return_value=95.1234) 
        self.my_label = MagicMock()
        self.input_window = MagicMock()  # Mock window


        self.entries = {
            'Pclass': MagicMock(get=MagicMock(return_value='1')),
            'Sex': MagicMock(get=MagicMock(return_value='male')),
            'Age': MagicMock(get=MagicMock(return_value='25')),
            'SibSp': MagicMock(get=MagicMock(return_value='0')),
            'Parch': MagicMock(get=MagicMock(return_value='0')),
            'Fare': MagicMock(get=MagicMock(return_value='50.0')),
            'Embarked': MagicMock(get=MagicMock(return_value='S'))
        }

    def test_tester_modele(self):
         # I call the function tester_modele with the mock objects and I simulate the test result
        tester_modele(self.mlp, self.testdata, self.my_label)

        self.testdata.assert_called_once_with(self.mlp)

        # Verifies that the label was updated with the test result
        expected_result_text = "Test result: 95.12% accuracy"
        self.my_label.configure.assert_any_call(text=expected_result_text)

    @patch('Code.input.titanic.prep_data.test_user_input')
    @patch('Code.input.titanic.prep_data.scaler')
    def test_afficher_donnees_survived(self, mock_test_user_input, mock_scaler):
        # Set the prediction to 1 (he survived)
        mock_test_user_input.return_value = 1

        # Set the prep.data.scaler to return a scaler object
        mock_scaler.fit = MagicMock()

        # Loop too much so dont work
        afficher_donnees(self.entries, self.mlp, self.my_label, self.input_window)

        # Check the label text for correct output
        expected_text = (
            "Entered data:\nPclass: 1\nSex: male\nAge: 25.0\n"
            "SibSp: 0\nParch: 0\nFare: 50.0\nEmbarked: S\n"
            "Prediction: Passenger survived"
        )
        self.my_label.configure.assert_called_with(text=expected_text)
        
        # To be sure that the window was destroyed
        self.input_window.destroy.assert_called_once()

if __name__ == '__main__':
    unittest.main()