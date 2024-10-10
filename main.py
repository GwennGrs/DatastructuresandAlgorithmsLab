def main():
    # Import the necessary modules
    from Code.NeuralNetwork.multilayerPerceptron import MLP
    from Code.input.titanic.prep_data import prep_traindata, testdata
    from Code.interface_package.interface import app_interface  # Call your interface

    # Preparing data for MLP model
    X_train, y_train = prep_traindata()

    # The number of outputs must be 3 (one-hot encoding for classes 1, 2, 3)
    mlp = MLP(num_inputs=X_train.shape[1], hidden_layers=[3, 3], num_outputs=2)  # 2 exits for classes 0 (not survived) and 1 (survived)
    # Training the MLP model
    mlp.train(X_train, y_train, epochs=400, learning_rate=0.1)    

    # Launch the interface with the MLP model and the testdata function
    app_interface(mlp, testdata)

if __name__ == "__main__":
    main()