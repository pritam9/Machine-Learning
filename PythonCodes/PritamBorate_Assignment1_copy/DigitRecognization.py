# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 20:21:16 2017

@author: Pritam
"""

import pandas as pd
import numpy as np
import math
from scipy.sparse import *
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from pandas import DataFrame

#Reading the train and test data from the file and storing it in an object
trainData = pd.read_csv('/MS_CS/Machine Learning/Assignment 1/ITCS6156_SLProject/DigitRecognition/optdigits_raining.csv', sep=',',header=None,
                        keep_default_na=False)

testData = pd.read_csv('/MS_CS/Machine Learning/Assignment 1/ITCS6156_SLProject/DigitRecognition/optdigits_test.csv', sep=',',header=None)

trainData.isnull().sum().sum()

df = pd.DataFrame(data=0, index = np.arange(3823), columns = np.arange(64))
dfLabels = pd.DataFrame(data=0, index = np.arange(3824), columns = np.arange(1))
Y= trainData[64]
trainData[1][2]
trainData.set_value(65,0,1)
#Creating sparse matrix for training data
    
#Using Decision Tree classifier to model the data
trainData[0:63]
trainData1=trainData.transpose()
trainData2=trainData1[0:63]
trainData3=trainData2.transpose()
model = DecisionTreeClassifier(max_depth=100)
#model.fit(df,trainLabels)
model.fit(trainData3,Y)
print("Model is -")
print("Prediction is - ")

#Trimming the test data contents and freeing the memory
ef=pd.DataFrame(data=0, index = np.arange(1797), columns = np.arange(64))
Z=testData[64]
testData1=testData.transpose()
testData2=testData1[0:63]
testData3=testData2.transpose()
#Using Decision Tree classifier to predict

prediction=model.predict(testData3)
print(prediction)

#Accuracy calculation
import numpy as np
from sklearn.metrics import accuracy_score
print(accuracy_score(prediction,Z))