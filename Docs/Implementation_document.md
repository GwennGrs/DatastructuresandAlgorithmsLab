# Implementation document

## Structure of the Program: Multilayer Perceptron (MLP)

The program is primarily a multi-layered neural network algorithm for classifications. 
[The MLP](https://github.com/GwennGrs/DatastructuresandAlgorithmsLab/blob/main/Code/NeuralNetwork/multilayer_perceptron.py) class is mainly composed of methods necessary for its learning and use. On y retrouve principalement les méthodes de Forward Propagation et Backward Propagation nécesaire au Train.
For this, [the main mathematical functions](https://github.com/GwennGrs/DatastructuresandAlgorithmsLab/blob/main/Code/NeuralNetwork/function.py) were implemented and put in another file. 
For the [extraction of the data and the preparation](https://github.com/GwennGrs/DatastructuresandAlgorithmsLab/blob/main/Code/input/titanic/prep_data.py) of the data in order to make the model efficient and optimized, this is done in another file containing all the methods necessary for data processing for training and testing from the raw data. 
[An interface](https://github.com/GwennGrs/DatastructuresandAlgorithmsLab/blob/main/Code/interface_package/interface.py) was also implemented in order to make the project easier to use and more aesthetic. This interface offers many methods such as being able to predict with our own data or even testing the accuracy of our model.

## Time and complexity 
To find the complexity it is necessary to decompose the program, in particular the main class (MLP), to see the different methods.
The complexity of the code lies mainly in the train function.

**The train method** starts with a loop of "epochs" times == E with another loop of the len of inputs. == I
Temporal: 0() = ExI
Spatial: O(1)

**The forward propagate method now:**
Time complexity: O(L x n^2) (for a network with L layers and n neurons per layer approximately).
Spatial complexity: O(L×n)O(L \times n)O(L×n)

**The Backward Propagation method:**
    L: number of layers in the neural network (given by len(self.derivatives)).
    ni: number of neurons in layer iii.
    d: dimension of the input (number of features of the inputs).
O(i=1∑Lb×ni−1×ni) is the total time complexity over all L layers, mainly due to matrix multiplication.
Time complexity:
O(L×b×n^2) if we assume n is the average number of neurons in each layer.

Spatial complexity:
If we simplify by assuming that each layer has n neurons, the spatial complexity becomes:
O(L*n)

**For gradient descent:**
Time complexity: O(L×n^2) (where L is the number of layers and n is the number of neurons per layer).
Spatial complexity: O(L×n^2), because the memory is dominated by the weights and derivatives of the layers.

**Overall complexity:**
Given that for each training the complexity of the gradient descent dominates greatly with its L*n^2. Simplifying, we arrive at a total complexity of O(E×N×L×n^2 ) where:
- E is the number of epochs,
- N is the number of training examples,
- L is the number of layers in the network,
- n is the average number of neurons per layer.

## Potential shortcomings and suggested improvements of the work.
I could have tried to improve the accuracy of my model. By progressively modifying my learning rate for example. Starting with a learning rate close to 1 and gradually decreasing.
I could have made the model adaptive for different datasets. The user could have entered his own dataset and the model would have determined by itself which configurations to use.

## Use of extensive language models (ChatGPT, etc.) 
I used ChatGPT to unblock me during technical problems, or to help me realize the interface by linking it to methods due to package and module problems.
Copilot helped me generate test datasets and test variables for my unit tests.

## References
The dataset : https://www.kaggle.com/c/titanic/data

Wikipedia of the MLP : https://fr.wikipedia.org/wiki/Perceptron_multicouche

For the comprehension of the Forward et Backward propagation : https://www.youtube.com/watch?v=99CcviQchd8&pp=ygUnZm9yd2FyZCBwcm9wYWdhdGlvbiBhbmQgYmFja3Byb3BhZ2F0aW9u

Another for the comprehension of the forward propagation : https://www.youtube.com/watch?v=9Ym1gtGat08&t=1061s&pp=ygUTZnJvd2FyZCBwcm9wYWdhdGlvbg%3D%3D

For the sigmoid function and these properties: https://en.wikipedia.org/wiki/Sigmoid_function

For the implementation from scratch of an MLP algorithm: https://www.kaggle.com/code/vitorgamalemos/multilayer-perceptron-from-scratch

For the basics of how to implement NN algorithms : https://www.youtube.com/watch?v=cAkMcPfY_Ns&t=349s

For the perceptron comprehension and implementation : https://www.youtube.com/watch?v=kft1AJ9WVDk

Help with implementing the model from scratch : https://www.youtube.com/watch?v=Py4xvZx-A1E

