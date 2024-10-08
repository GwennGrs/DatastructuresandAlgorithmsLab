import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

scaler = MinMaxScaler()
encoder = OneHotEncoder(sparse_output=False)

def prep_traindata():
    """
    Prepares the training data for the MLP model.
    This function reads the training data from a CSV file, selects relevant columns,
    handles missing values, converts categorical columns to numerical, normalizes
    the features, and applies one-hot encoding to the target variable.
    Returns:
        tuple: A tuple containing:
            - X_train (numpy.ndarray): The normalized feature matrix for training.
            - y_train (numpy.ndarray): The one-hot encoded target matrix for training.
    """ 
    train = pd.read_csv('Code/input/titanic/train.csv')
    # BEGIN: Data preparation for the MLP model
    # Select relevant columns and handle missing values
    train = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    train = train.dropna()

    # Convert categorical columns to numerical
    train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
    
    # Convert 'Embarked' column to numerical
    train['Embarked'] = train['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

    # Separate features (X) and target (y)
    X_train = train.drop('Survived', axis=1)
    y_train = train['Survived']

    # Normalize features
    X_train = scaler.fit_transform(X_train)

    # Apply one-hot encoding to the target (binary in this case)
    y_train = encoder.fit_transform(y_train.values.reshape(-1, 1))
    # END: Data preparation for the MLP model
    return X_train, y_train

def testdata(mlp):
    """
    Prepares test data and evaluates the MLP model on it.
    This function reads the test data, selects the relevant columns,
    handles missing values, converts categorical columns to numeric,
    normalizes the features and applies one-hot encoding on the target variable.
    It then evaluates the provided MLP model on the test data and displays the accuracy.
    
    Args:
        mlp (object): The MLP model to test. The model must have a
                        `forward_propagate` method that takes the features as input
                        and returns the prediction.
    
    Returns:
        None
    """
    # Preparing test data for the MLP model
    test = pd.read_csv('Code/input/titanic/test_fusion.csv')
    test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]
    test = test.dropna()

    # Convert categorical columns to numerical
    test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
    
    # Convert 'Embarked' column to numerical
    test['Embarked'] = test['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

    # Separate features and target (Survived)
    X_test = test.drop('Survived', axis=1)
    y_test = test['Survived']

    # Normalize features
    scaler.fit_transform(X_test)
    X_test = scaler.transform(X_test)  # Normalize test data
    
    # Test the model on the test data
    correct = 0
    for i, input in enumerate(X_test):
        output = mlp.forward_propagate(input)
        predicted_class = np.argmax(output)  # Returns the predicted class (0 or 1)
        true_class = y_test.iloc[i]  # The true class is 0 or 1
        if predicted_class == true_class:
            correct += 1

    # Display the model's accuracy on the test set
    return(((correct / len(X_test)) * 100))

def test_user_input(mlp, inputs):
    """
    Tests the user input data on the trained MLP model.
    This function takes the user input data, preprocesses it, and evaluates
    the trained MLP model on the preprocessed data.
    Args:
        inputs (dict): A dictionary containing the user input data.
    Returns:
        float: The prediction accuracy of the model on the user input data.
    """
    print("Testing user input data...")

    if scaler.fit is None:
        raise ValueError("Scaler not fitted. Please train the model first.")
    
    # Convert categorical columns to numerical
    inputs['Sex'] = {'male': 0, 'female': 1}.get(inputs['Sex'], 0)
    inputs['Embarked'] = {'C': 0, 'Q': 1, 'S': 2}.get(inputs['Embarked'], 2)  # Default to 'S' if not found

    # Create a DataFrame from user inputs
    user_df = pd.DataFrame([inputs])

    # Reorder columns to match the training order
    user_df = user_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    
    # Normalize features
    X_user = scaler.transform(user_df)

    # Test the model on user data
    output = mlp.forward_propagate(X_user[0])
    predicted_class = np.argmax(output)  # Returns the predicted class (0 or 1)
    print(predicted_class)
    return predicted_class
