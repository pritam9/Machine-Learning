# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 15:05:56 2017

@author: Pritam
"""

#Importing the libraries
#The libraries used are 'pandas' and scikit-learn.
#Pandas is being used to import the dataset into a dataframe 
#scikit-learn provides the libraries to implement perceptron network model and 
#predictions over the test data.

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#Using pandas import the file in the form of dataframe. 
#In this scenario, the file is being stored in the same folder as the
#working directory
trainDigits = pd.read_csv('/MS_CS/Machine Learning/Assignment 1/ITCS6156_SLProject/DigitRecognition/optdigits_raining.csv',header=None)
testDigits = pd.read_csv('/MS_CS/Machine Learning/Assignment 1/ITCS6156_SLProject/DigitRecognition/optdigits_test.csv',header=None)

#The dataset is such that first 64 columns are the features and tha last column 
#is the digit which that row belongs to.
#Using ix the segregation is done
featuresTrain = trainDigits.ix[:,'0':'63']
featuresTest = testDigits.ix[:,'0':'63']


#The dataframe is converted into the matrix 
trainFeaturesMatrix = featuresTrain.as_matrix(columns=None)
testFeaturesMatrix = featuresTest.as_matrix(columns=None)


trainDataframe = trainDigits.ix[:,'64':'64']
testDataframe = testDigits.ix[:,'64':'64']

trainMatrix = trainDataframe.as_matrix(columns=None)
testMatrix = testDataframe.as_matrix(columns=None)

#The trained matrices are passed to the library  to create the model and 
#predict the outcome of the test data
#The function used is the sigmoid function. The learning rate and 
#regularization parameters are varied for empirical data
classifier = MLPClassifier(solver='adam',alpha=0.01,hidden_layer_sizes=(64),random_state=1,learning_rate_init=0.01)
classifier.fit(trainFeaturesMatrix, trainMatrix)

#Prediction
prediction=classifier.predict(testFeaturesMatrix)

print (accuracy_score(prediction, testMatrix))