# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 15:20:19 2017

@author: Pritam
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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
classifier = KNeighborsClassifier(n_neighbors=10, weights='distance')
classifier.fit(trainFeaturesMatrix, trainMatrix)

#Prediction
prediction=classifier.predict(testFeaturesMatrix)

print (accuracy_score(prediction, testMatrix))