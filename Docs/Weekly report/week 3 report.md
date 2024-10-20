# Weekly report 3
*6 hours*

## NN Implementation:
I implemented an MLP (multilayer perceptron) algorithm using the principles used for the creation of the individual neuron created later. I then reuse the sigmoid function and its derivative, the forward function allowing the propagation of inputs via the activation functions and the train method using backward propagation to correct the weights in order to adjust the model.
We lay the foundations of the network with the possibility of choosing the number of inputs, outputs and hidden layers.

How to choose the number of neurons and number of hidden layers?
*Choosing the number of neurons and the number of hidden layers in a neural network (MLP) is a crucial step, but there is no exact or universal formula for this. The choice depends on several factors, including the complexity of the problem, the size of the data, and the type of task (classification, regression, etc.).*

## Model structure:
I created the MLP class to represent the structure of the model. The parameters to enter are therefore the number of inputs, the number of layers and the number of neurons as well as the number of desired output values. I set unimportant values by default.

## Documentation:
I started to write the documentation of my algorithm with the functions and principles used in my MLP algorithm. Including explanations on the weights and their operation, the choice of the number of inputs, the number of layers and the number of neurons per layer.

So I did a lot of theory on the operation of forward propagation, backward propagation and the need for non-linearity provided by the sigmoid function.
*(the documentation is draft at the moment it is temporary)*

## First test of the algorithm:
So I created a jupyter notebook file to test my algorithm. I used the Iris dataset provided as standard by the sklearn module (temporary). I then adjusted the number of neurons, layers and tests to have the most efficient model with the best ratio (temp/precision).
I arrive at an average precision of about 97% which seems satisfactory to me for the moment.
## Other tasks:
I installed and tested the different tools to test the quality of the code as well as the coverage of the tests.
*(Pylint and Coverage)*
I also thought about using a python module to save my model once trained instead of retraining it each time I restart the program but I don't know if this is a good idea.
## For the next week 
For week 4, I will link my algorithm to my class so that I can enter the data and have a more pleasant and aesthetic use of the algorithm. I will also start writing tests and correct what is needed to make my code cleaner.

I will also use another dataset with more data and more features in order to have a more complete model.
