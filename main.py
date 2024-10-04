def main():
    # Importer les modules nécessaires
    from Code.NeuralNetwork.multilayerPerceptron import MLP
    from Code.input.titanic.prep_data import prep_traindata, testdata
    from Code.interface_package.interface import app_interface  # Appeler ton interface

    # Préparation des données pour le modèle MLP
    X_train, y_train = prep_traindata()

    # Le nombre de sorties doit être 3 (encodage one-hot pour les classes 1, 2, 3)
    mlp = MLP(num_inputs=7, hidden_layers=[3,4], num_outputs=3)
    
    # Entraîner le modèle
    mlp.train(X_train, y_train, epochs=250, learning_rate=0.1)

    # Lancer l'interface avec le modèle MLP et la fonction testdata
    app_interface(mlp, testdata)

if __name__ == "__main__":
    main()
