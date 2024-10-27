# Implementation of the NN:
The foundations of the network are laid with the possibility of choosing the number of inputs, outputs and hidden layers.

How to choose the number of neurons and number of hidden layers?
*The choice of the number of neurons and the number of hidden layers in a neural network (MLP) is a crucial step, but there is no exact or universal formula for this. The choice depends on several factors, including the complexity of the problem, the size of the data, and the type of task (classification, regression, etc.).*

### For the choice of neurons: 
If you have n features as input and m classes as output, you can choose a number of neurons in the first hidden layer that lies between n and m. This could look like a gradual reduction in the number of neurons in each layer.

To find out which is the most optimal, test different numbers of neurons and layers.

You still need to be careful about **over** and **underlearning** if you choose the wrong number. You also need to think about **cost**.

### Model structure:
I've created the MLP class to represent the structure of the model. The parameters to be entered are the number of inputs, the number of layers and the number of neurons, as well as the number of desired output values. I've set the default values to unimportant.

#### Weight:
Weights are chosen at random to start with, as it's impossible to set any useful value because it's impossible to predict beforehand.

However, the number of weights and the matrix format are predictable. For a model with 4 inputs, 1 layer of 5 neurons and 3 outputs, the format is [4,5,3]. We end up with a matrix [4,5] for the activation matrix of the layer between the input layer and the hidden layer, then a matrix [5,3] for the matrix enabling us to move from the hidden layer to the output layer. Obviously, the weights are between 0 and 1. 

#### Derivatives :
For the backpropagation phase, we create a matrix of the same size as the weight matrices, which will later be used to store the weight gradients. It is filled with 0 using the np.zeros() method. 

#### Activation :
For forward propagation, we'll need the outputs of each neuron at each layer. These arrays will contain the outputs of the neurons in a layer after application of the activation method (in this case, the sigmoid function). The first array will be the inputs and the last ones will be the outputs.

#### Forward propagation : 
See how forward propagation works. 
https://www.youtube.com/watch?v=9Ym1gtGat08
You call the forward_propagate method, which calculates the output of the network. This process passes the input data through all layers of the neural network, applying weights and activation functions (in this case, a sigmoid function). The output of each layer becomes the input for the next layer. 
Finally, the final output is calculated and stored.

**Input**: an array of data inputs.
**Output**: an array with output values after activation.

We use the np.dot function to produce a matrix product between the weight matrices and the matrices (array i.e. matrix 1,n).
If the result of the sigmoid function is > 0.5, the result is 1, otherwise 0.

#### Back propagation:
We use the derivative of the Sigmoid because it was our activation function. If we'd used another function, we'd have taken its derivative.
Summary of the Backpropagation Process
1.	Initialization: The error of the output layer is multiplied by the derivative of the activation function to obtain delta.
2.	Derivative calculation : Weight derivatives for each layer are calculated by multiplying previous activations by delta_re.
3.	Error propagation: The error is propagated backwards through the network to adjust the weights of previous layers.
4.	Derivative storage: Calculated derivatives are stored for later use in updating weights.

#### Advanced:
_With an old dataset_

The MLP code for the IRIS dataset works well.
1 layer of 5 neurons is the most optimal for the current dataset.
I used the default IRIS dataset from scikit.
Depending on the number of loops for the train, the results may be better or worse. Depending on the number of data loops and the time it takes. If for 2 seconds of train, you get an average of 80% accuracy, however for 10sec you get 96% on average, so you have to choose between sacrificing performance or not for a little better accuracy.
The most opti I've found is 250 loop with an average accuracy of 98%.

The code network.py is a demonstration of the structure of my neural network.