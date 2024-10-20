# 5st Weekly report :
*11 hours*

## Important changes
This week I made many changes to the final version of my project. I first removed a layer of neurons for my model because the accuracy was too volatile, due to overtraining of the data due to an excessive number of neurons and layers.
I also realized that using p_dummies for the Embarked variable was a bad choice, as it added variables that were not needed and hampered the functioning of my model.

## Change target variable
During my training I trained my model to predict the passenger's P_class, which was not very useful and interesting.
The test file linked to the train file did not have a Survived variable while the train file did.
Finally, a file named gender_submission (present in my inputs/titanic folder) had the variable Survived associated with a passenger ID. I used panda's merge function to merge the test file with the gender_submission file (new file is called test_fusion). I was then able to change the target variable, taking Survived as the target, which makes more sense.
*link(https://www.kaggle.com/code/enigmak/tabnet-deep-neural-network-for-tabular-data/input?select=train.csv)*

## Around the project
I translated and added comments where they were missing and made the missing docstrings.
The various files "Userguide", "Implementation document" and "Specification Document" have been updated and/or completed.
For dependency issues and installing the necessary modules I implemented poetry on my project to make it easier to use (some problem with the scikit-learn module which uses Joblib and which does not want to install).

## Useful addition to the project
I added some methods necessary for the operation of the project, especially the interface. For example, a function allowing to use data entered by the user in the model (this implies having to normalize them with the complete dataset so I used scaler.fit).
A progress bar in the terminal was added in order to follow the progress of our model over time and to be sure that there are no problems.
I also added exception handling in case the data entered by the user is not of the expected type.

## For the next week 
For the last week, I only have to finish writing the important documents for the description of the program and its use. I also have some tests to do other than the unit tests.