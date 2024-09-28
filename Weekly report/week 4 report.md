# 1st Weekly report :
*5 hours*

## Change Dataset
For week 4, I mainly wanted to check the reliability of my model. For this I looked for several tabular datasets with more variables than a simple iris dataset.
I then found a dataset on Kaggle that allows me to determine the class of a passenger using 7 variables.
*(I found another dataset about adults but I didn't use it.)*

## Prepare the dataset
Once the dataset in csv format was imported using the python pandas module. I had to perform various manipulations on the dataset in order to make it usable and more optimized for the model (as for the Iris dataset).

I first had to transform the **qualitative variables** into **quantitative ones** to be able to pass them into the NN.
I then **normalized** my data, taking care to remove the class to be predicted beforehand (P_class).
Finally I apply the **One Hot encoded** method that I previously coded for the variable to be predicted.

I did this whole process for both training and testing data.

## Model Reviews
Finally, the model applied to the data with a configuration of type [7,3,4]. I then tested different configurations and it turned out that one of the most optimized is one using 2 layers: [7,3,4,3] arriving at an accuracy of 87%. This allowed me to see that the backpropagation method works correctly.

I can also add layers for equal precision but that would be meaningless because it requires more resources for the same result.

## Do some test
I then carried out tests for the coverage of my code with Unitest.
The function.py and MLP.py files have been tested correctly, the only one not tested is single_neuron.py because all its methods are in the function.py file now to facilitate the use of functions instead of rewriting it in all the files using it.

## For the next week 
For next week, I would like to link the use of the interface with the MLP class, either by entering data on a model previously trained by a defined dataset or by being able to add a dataset ourselves which would be processed, trained and then made available to the user.

I will also have to translate the comments into English and start writing the "how to" in the ReadME.md.

I will also have to rework my One Hot encoding function to see what differs from sklearn's.