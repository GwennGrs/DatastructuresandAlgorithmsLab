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
    # BEGIN: Préparation des données pour le modèle MLP
    # Sélectionner les colonnes pertinentes et gérer les valeurs manquantes
    train = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    train = train.dropna()

    # Convertir les colonnes catégorielles en numériques
    train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
    
    # Convertir les colonnes 'Embarked' en numérique
    train['Embarked'] = train['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

    # Séparer les caractéristiques (X) et la cible (y)
    X_train = train.drop('Survived', axis=1)
    y_train = train['Survived']

    # Normaliser les caractéristiques
    X_train = scaler.fit_transform(X_train)

    # Appliquer one-hot encoding à la cible (binaire dans ce cas)
    y_train = encoder.fit_transform(y_train.values.reshape(-1, 1))
    # END: Préparation des données pour le modèle MLP
    return X_train, y_train

def testdata(mlp):
    """
    Prépare les données de test et évalue le modèle MLP sur celles-ci.
    Cette fonction lit les données de test, sélectionne les colonnes pertinentes,
    gère les valeurs manquantes, convertit les colonnes catégorielles en numériques,
    normalise les caractéristiques et applique un one-hot encoding sur la variable cible.
    Elle évalue ensuite le modèle MLP fourni sur les données de test et affiche la précision.
    
    Args:
        mlp (object): Le modèle MLP à tester. Le modèle doit avoir une méthode
                      `forward_propagate` qui prend en entrée les caractéristiques
                      et retourne la prédiction.
    
    Returns:
        None
    """
    # Préparation des données de test pour le modèle MLP
    test = pd.read_csv('Code/input/titanic/test_fusion.csv')
    test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]
    test = test.dropna()

    # Convertir les colonnes catégorielles en numériques
    test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
    
    # Convertir les colonnes 'Embarked' en numérique
    test['Embarked'] = test['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

    # Séparer les caractéristiques et la cible (Survived)
    X_test = test.drop('Survived', axis=1)
    y_test = test['Survived']

    # Normaliser les caractéristiques
    scaler.fit_transform(X_test)
    X_test = scaler.transform(X_test)  # Normaliser les données de test
    
    # Tester le modèle sur les données de test
    correct = 0
    for i, input in enumerate(X_test):
        output = mlp.forward_propagate(input)
        predicted_class = np.argmax(output)  # Retourne la classe prédite (0 ou 1)
        true_class = y_test.iloc[i]  # La vraie classe est 0 ou 1
        if predicted_class == true_class:
            correct += 1

    # Afficher la précision du modèle sur l'ensemble de test
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
    
    # Convertir les colonnes catégorielles en numériques
    inputs['Sex'] = {'male': 0, 'female': 1}.get(inputs['Sex'], 0)
    inputs['Embarked'] = {'C': 0, 'Q': 1, 'S': 2}.get(inputs['Embarked'], 2)  # Par défaut 'S' si non trouvé

    # Créer un DataFrame à partir des entrées utilisateur
    user_df = pd.DataFrame([inputs])

    # Réorganiser les colonnes pour correspondre à l'ordre de l'entraînement
    user_df = user_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    
    # Normaliser les caractéristiques
    X_user = scaler.transform(user_df)

    # Tester le modèle sur les données utilisateur
    output = mlp.forward_propagate(X_user[0])
    predicted_class = np.argmax(output)  # Retourne la classe prédite (0 ou 1)
    print(predicted_class)
    return predicted_class
