import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
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
    # BEGIN: Préparation des données pour le modèle MLP
    # Sélectionner les colonnes pertinentes et gérer les valeurs manquantes
    train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    train = train.dropna()

    # Convertir les colonnes catégorielles en numériques
    train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
    train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)

    # Normaliser les caractéristiques
    X_train = scaler.fit_transform(train.drop('Pclass', axis=1))  # Supposons que 'Pclass' est la cible pour cet exemple

    # Appliquer one-hot encoding à la cible
    y_train = encoder.fit_transform(train[['Pclass']])
    # END: Préparation des données pour le modèle MLP
    return X_train, y_train

def testdata(mlp):
    """
    Prepares the test data and evaluates the MLP model on it.
    This function reads the test data from a CSV file, selects relevant columns,
    handles missing values, converts categorical columns to numerical, normalizes
    the features, and applies one-hot encoding to the target variable. It then
    evaluates the provided MLP model on the test data and prints the accuracy.
    Args:
        mlp (object): The MLP model to be tested. The model should have a method
                    `forward_propagate` that takes an input and returns the output.
    Returns:
        None
    """
    # Préparation des données de test pour le modèle MLP
    test = pd.read_csv('Code/input/titanic/test.csv')
    test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    test = test.dropna()

    # Convertir les colonnes catégorielles en numériques
    test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
    test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)

    # Normaliser les caractéristiques
    X_test = scaler.transform(test.drop('Pclass', axis=1))  # Supposons que 'Pclass' est la cible pour cet exemple
    # Appliquer one-hot encoding à la cible
    y_test = encoder.transform(test[['Pclass']])

    # Tester le modèle sur les données de test
    correct = 0
    for i, input in enumerate(X_test):
        output = mlp.forward_propagate(input)
        predicted_class = np.argmax(output)
        true_class = np.argmax(y_test[i])
        if predicted_class == true_class:
            correct += 1

    return("Précision sur l'ensemble de test: {:.2f}%".format((correct / len(X_test)) * 100))