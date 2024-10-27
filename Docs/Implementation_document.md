# Implementation document

## Structure of the Program: Multilayer Perceptron (MLP)

The program is primarily a multi-layered neural network algorithm for classifications. 
[The MLP](https://github.com/GwennGrs/DatastructuresandAlgorithmsLab/blob/main/Code/NeuralNetwork/multilayer_perceptron.py) class is mainly composed of methods necessary for its learning and use. On y retrouve principalement les méthodes de Forward Propagation et Backward Propagation nécesaire au Train.
For this, [the main mathematical functions](https://github.com/GwennGrs/DatastructuresandAlgorithmsLab/blob/main/Code/NeuralNetwork/function.py) were implemented and put in another file. 
For the [extraction of the data and the preparation](https://github.com/GwennGrs/DatastructuresandAlgorithmsLab/blob/main/Code/input/titanic/prep_data.py) of the data in order to make the model efficient and optimized, this is done in another file containing all the methods necessary for data processing for training and testing from the raw data. 
[An interface](https://github.com/GwennGrs/DatastructuresandAlgorithmsLab/blob/main/Code/interface_package/interface.py) was also implemented in order to make the project easier to use and more aesthetic. This interface offers many methods such as being able to predict with our own data or even testing the accuracy of our model.

## Time and complexity 

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
