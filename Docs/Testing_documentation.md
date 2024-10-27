# Testing documentation

In this document you will find various information on the tests related to this project.

## The coverage report of the unit tests.

| File                                          | Statements | Missing | Coverage | Missing Lines   |
|-----------------------------------------------|------------|---------|----------|-----------------|
| `Code\NeuralNetwork\__init__.py`              | 0          | 0       | 100%     |                 |
| `Code\NeuralNetwork\function.py`              | 14         | 6       | 57%      | 58-65          |
| `Code\NeuralNetwork\multilayer_perceptron.py` | 62         | 0       | 100%     |                 |
| `Code\input\titanic\__init__.py`              | 0          | 0       | 100%     |                 |
| `Code\input\titanic\prep_data.py`             | 46         | 1       | 98%      | 124            |
| `Code\test\__init__.py`                       | 0          | 0       | 100%     |                 |
| `Code\test\test_integration.py`               | 15         | 1       | 93%      | 40             |
| **TOTAL**                                     | **137**    | **8**   | **94%**  |                 |



My **prep_data** and **interface**_(for 2 methods)_ files could not be tested properly because they required the use of my model, so the unit tests required the use of a MagicMock. In addition, the use of the latter most of the time did not match my MLP model.

## What has been tested and how? What kind of inputs were used for the testing? 

**The function methods :** [function.py](https://github.com/GwennGrs/DatastructuresandAlgorithmsLab/blob/main/Code/NeuralNetwork/function.py) 

In the test_function.py file I tested all the methods of the function.py file individually.

The sigmoid and sigmoid_derivative functions being mathematical functions I just checked that with a given value I got the expected value by checking myself. The inputs were classic numbers (possibly with decimal point and/or negative).

For the one_hot_encoding function I only compared if the tables received as output were the expected ones. The inputs were a data table between 0 and n-1 (n being the number of possible classes).

For the mse error function, I compared the sigmoid and sigmoid_derivatives functions but this time on tables of numbers. I performed several tests to be sure that most of the possible configurations were functional.

**The multilayer perceptron methods :** [multilayer_perceptron.py](https://github.com/GwennGrs/DatastructuresandAlgorithmsLab/blob/main/Code/NeuralNetwork/multilayer_perceptron.py)

To test all my methods on the MLP model, it was necessary to create a model. For this I created a model with the 3 neurons in the input layer, a layer of 3 and 5 neurons for the hidden layer and 2 neurons for the output layer.

I also redefined the weights myself in order not to have this random side linked to the random generation of the latter.

The first test for the constructor simply consists of checking whether the variables num_inputs, hidden_layers and num_output have the right values.

A fictitious dataset is created for the forward_propagate and backward_propagate functions. With these 2 datasets I test if after executing the 2 methods the outputs have the right number. For the backward_propagate function it was more complex to test, so I take care of checking that the derivatives are correctly calculated and that they are well modified.

Finally, I test the train method on a fictitious dataset and the initialization of a simple model in order to test whether the weights are indeed modified in terms of this method.

**The interface method :** [interface.py](https://github.com/GwennGrs/DatastructuresandAlgorithmsLab/blob/main/Code/interface_package/interface.py)

Testing the interface was complex because it was difficult to check whether a display was correct or not. K
I then used the MagickMock module which allows you to simulate complex structures.

For the first method, which is tester_modele I simulated the entire creation of an interface using Mock objects. This method using the result of my **testdata** function, I then simulated an abstract result so as not to have to execute the method in order to simplify the test. Finally, I compared the expected outputs and those received.

For the method affiche_donnees, I simulated an individual in the self using Mock and using the patch of the MagickMock module I simulated the results and the structure of test_user_input and the scaler. The result being set to 1 and the inputs I knew in advance what the function should write. I then compared the expected writing and the one received.

**The integration test :** [test_integration.py](https://github.com/GwennGrs/DatastructuresandAlgorithmsLab/blob/main/Code/test/test_integration.py)

For this test I simulated the overall and complete operation of my model. I then started by launching the prep_traindata method which, using the train.csv file, retrieves the Kaggle data on the Titanic.

I then instantiated the model on the same bases as the main one. I trained it with the data returned by pre_traindata.

Then by taking an individual in the dataset I transformed it into input and checked if the model predicted the right class.
The last check was to evaluate the model globally, for this I checked if the overall accuracy of the model was greater than 80% using the testdata method.

## How can the tests be repeated?

You can easily repeat these tests. 
To do this, place yourself in the root and execute the command:
```bash
coverage run -m unittest .\Code\test\"name-of-the-test".py
```

And if you want to see the coverage, go to the test folder :
```bash
cd .\Code\test\
python -m coverage report
```