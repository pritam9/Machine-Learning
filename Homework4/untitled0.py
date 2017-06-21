# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:11:53 2017

@author: Pritam
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 16:11:41 2017

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
from sklearn.svm import SVC

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

print("Model is -")
print("Prediction is - ")

#Trimming the test data contents and freeing the memory
ef=pd.DataFrame(data=0, index = np.arange(1797), columns = np.arange(64))
Z=testData[64]
testData1=testData.transpose()
testData2=testData1[0:63]
testData3=testData2.transpose()
#Using Decision Tree classifier to predict
model = DecisionTreeClassifier(max_depth=16)
#model.fit(df,trainLabels)
model.fit(trainData3,Y)
prediction=model.predict(testData3)
print(prediction)

#Accuracy calculation
import numpy as np
from sklearn.metrics import accuracy_score
print("Accuracy with Decision Tree..")
print(accuracy_score(prediction,Z))
print("With SVM...")
model = SVC(kernel='linear', probability=False, C=0.001)
model.fit(trainData3,Y)
print(model)
prediction=model.predict(testData3)
print("Accuracy After Boosting..")
print(accuracy_score(prediction,Z))