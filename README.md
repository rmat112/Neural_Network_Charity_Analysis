# Neural_Network_Charity_Analysis

## Overview of Analysis
With my knowledge of machine learning and neural networks, I have used the features in the provided dataset ([charity_data.csv](https://github.com/rmat112/Neural_Network_Charity_Analysis/blob/main/charity_data.csv)) to help create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

Alphabet Soup’s business team provided a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

## Process Description
This section describes the steps taken to complete this challenge.

### 1. Preprocessing Data for a Neural Network Model
The following preprocessing steps were performed:<br/>
Notebook: [AlphabetSoupCharity.ipynb](https://github.com/rmat112/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb)
- The EIN and NAME columns were dropped
- The columns with more than 10 unique values were grouped together
- The categorical variables were encoded using one-hot encoding
- The preprocessed data was split into features and target arrays
- The preprocessed data was split into training and testing datasets
- The numerical values were standardized using the StandardScaler() module

### 2. Compile, Train, and Evaluate the Model
The neural network model using Tensorflow Keras contains working code that performs the following steps:<br/>
Notebook: [AlphabetSoupCharity.ipynb](https://github.com/rmat112/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb)
- The number of layers, the number of neurons per layer, and activation function are defined
- An output layer with an activation function is created
- There is an output for the structure of the model
- There is an output of the model’s loss and accuracy
- The model's weights are saved every 5 epochs
- The results are saved to an HDF5 file: [AlphabetSoupCharity.h5](https://github.com/rmat112/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.h5)

### 3. Optimize the Model
Three attempts were made to optimize the model in order to increase predictive accuracy. Desired acccuracy of 75% could not be achieved due to time constraint. however, suggestions are made for further work on the model.<br/>
Notebook: [AlphabetSoupCharity_Optimization.ipynb](https://github.com/rmat112/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.ipynb)
- Working code is provided that makes three attempts to increase model performance using the following steps:
- An attempt is made to remove noisy variables from features
- Additional neurons are added to hidden layers
- Additional hidden layers are added
- The activation function of hidden layers or output layers is changed for optimization
- The model's weights are saved every 10 epochs
- The results are saved to an HDF5 file: [AlphabetSoupCharity_Optimization.h5](https://github.com/rmat112/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.h5)

## Results
### Data Preprocessing
- The IS_SUCCESSFUL column is considered the target for my model.
- The APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATION, and ASK_AMT columns are considered to be the features of my model.
- The EIN, NAME columns are neither targets nor features and hence were removed from the input data.

### Compiling, Training, and Evaluating the Model
- During the 3rd and final optimization effort I selected 100 neurons with a tanh function for my first layer, 30 nuerons with a tanh function for the second, 10 neurons with a sigmoid function for the third, and a sigmoid function for the outer layer. These were different from the activation functions that I used during my 2nd optimization in an attempt to increase the accuracy.
- With 3 optimization efforts, I was only able to achive 72.9% accuracy which is lower than the target model performance of 75%.
- In order to achieve target model performance, I tried the following:
    - 1st optimization: Remove noisy variables are removed from features
    - 2nd optimization: Adding more neurons to to hidden layers and adding a third hidden layer
    - 3rd optimization: Changing the activation functions of hidden layers
    
## Summary
- The first model with 2 hidden layers yielded an accuracy of 72.89% and a loss of 0.559.
- The second model (first optimization) included fewer features and no other changes. This model yielded an accuracy of 72.26% and a loss of 0.569
- The third model (2nd optimization) included the same number of features as the first model. There were 3 hidden layers and increased number of neurons in the first layer. All hidden layers used relu activation fubctions and the output layer used sigmoid activation function. his model yielded an accuracy of 73.00% and a loss of 0.560
- The fourth model (3rd optimization) used tanh activation functions for first two hidden layers and sigmoid function for the third and output layers. This model yielded an accuracy of 72.99% and a loss of 0.561.
- The results of this challenge indicate that changing different parameters did not have any significant change in model performance. This indcates that a neural network model might not be the best solution for this dataset. 

I would like to try the RandomForest model and see if it performs better. Random forest algorithms are beneficial because they:

1. Are robust against overfitting as all of those weak learners are trained on different pieces of the data.
2. Can be used to rank the importance of input variables in a natural way.
3. Can handle thousands of input variables without variable deletion.
4. Are robust to outliers and nonlinear data.
5. Run efficiently on large datasets.
6. Provide results in significantly less time as compared to a neural network model.


