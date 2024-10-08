import unittest
from unittest.mock import MagicMock, patch
from Code.interface_package.interface import afficher_donnees
from Code.input.titanic.prep_data import test_user_input

class TestInterface(unittest.TestCase):

    def setUp(self):
        self.entries = {
            'Pclass': MagicMock(),
            'Sex': MagicMock(),
            'Age': MagicMock(),
            'SibSp': MagicMock(),
            'Parch': MagicMock(),
            'Fare': MagicMock(),
            'Embarked': MagicMock()
        }
        self.mlp = MagicMock()
        self.my_label = MagicMock()

    @patch('Code.input.titanic.prep_data.scaler')
    def test_afficher_donnees_survived(self, mock_scaler):
        # Mocking the entries
        self.entries['Pclass'].get.return_value = '1'
        self.entries['Sex'].get.return_value = 'female'
        self.entries['Age'].get.return_value = '29'
        self.entries['SibSp'].get.return_value = '0'
        self.entries['Parch'].get.return_value = '0'
        self.entries['Fare'].get.return_value = '100'
        self.entries['Embarked'].get.return_value = 'C'

        # Mocking the test_user_input function to return 1 (survived)
        test_user_input.return_value = 1

        # Mocking the scaler to avoid NotFittedError
        mock_scaler.transform.return_value = [[1, 0, 29, 0, 0, 100, 1]]

        afficher_donnees(self.entries, self.mlp, self.my_label)

        expected_text = ("Entered data:\nPclass: 1\nSex: female\nAge: 29\n"
                         "SibSp: 0\nParch: 0\nFare: 100\nEmbarked: C\n"
                         "Prediction: Passenger survived")
        self.my_label.configure.assert_called_with(text=expected_text)

    @patch('Code.input.titanic.prep_data.scaler')
    def test_afficher_donnees_not_survived(self, mock_scaler):
        # Mocking the entries
        self.entries['Pclass'].get.return_value = '3'
        self.entries['Sex'].get.return_value = 'male'
        self.entries['Age'].get.return_value = '34.5'
        self.entries['SibSp'].get.return_value = '0'
        self.entries['Parch'].get.return_value = '0'
        self.entries['Fare'].get.return_value = '7.8292'
        self.entries['Embarked'].get.return_value = 'Q'

        # Mocking the test_user_input function to return 0 (didn't survive)
        test_user_input.return_value = 0

        # Mocking the scaler to avoid NotFittedError
        mock_scaler.transform.return_value = [[3, 1, 22, 1, 0, 7.25, 0]]

        afficher_donnees(self.entries, self.mlp, self.my_label)

        expected_text = ("Entered data:\nPclass: 3\nSex: male\nAge: 22\n"
                         "SibSp: 1\nParch: 0\nFare: 7.25\nEmbarked: S\n"
                         "Prediction: Passenger didn't survive")
        self.my_label.configure.assert_called_with(text=expected_text)

if __name__ == '__main__':
    unittest.main()