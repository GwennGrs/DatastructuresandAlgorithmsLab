import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from Code.input.titanic.prep_data import prep_traindata, testdata, test_user_input

# class TestPrepData(unittest.TestCase):

#     @patch('pandas.read_csv')
#     def test_prep_traindata(self, mock_read_csv):
#         # Mock the data returned by pd.read_csv
#         mock_data = pd.DataFrame({
#             'Survived': [1, 0, 1],
#             'Pclass': [3, 1, 2],
#             'Sex': ['male', 'female', 'female'],
#             'Age': [22, 38, 26],
#             'SibSp': [1, 1, 0],
#             'Parch': [0, 0, 0],
#             'Fare': [7.25, 71.2833, 7.925],
#             'Embarked': ['S', 'C', 'Q']
#         })
#         mock_read_csv.return_value = mock_data

#         X_train, y_train = prep_traindata()

#         # Check the shape of the returned arrays
#         self.assertEqual(X_train.shape, (3, 7))
#         self.assertEqual(y_train.shape, (3, 2))

#         # Check if the data is normalized
#         self.assertTrue(np.all(X_train >= 0) and np.all(X_train <= 1))

# if __name__ == '__main__':
#     unittest.main()