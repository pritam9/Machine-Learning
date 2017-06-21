# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 18:52:58 2017

@author: Pritam
"""

import csv
import numpy as np
from sklearn import tree
import math

with open('/MS_CS/Machine Learning/Assignment 1/ITCS6156_SLProject/DigitRecognition/optdigits_raining.csv') as trainingFile:
    reader = csv.reader(trainingFile)
    
    X=[]
    Y=[]
    
    for row in reader:
        X.append(row[:64])
        Y.append(row[64])
            
#num_correct = 0;
length_TrainingSet = len(X)

percentage_training = 0.7
    
len_train = math.floor(length_TrainingSet * percentage_training);
    
X_train = X[:len_train]
Y_train = Y[:len_train]
    
X_validation = X[len_train:len(X)]
Y_validation = Y[len_train:len(Y)]

    

clf = tree.DecisionTreeClassifier(max_depth=15);
clf = clf.fit(X_train, Y_train)
print("Done Classifying");
    
num_correct = 0;
for i in range(0,len(X_train)):
        output_predicted = clf.predict([X_train[i]])
        originalOutput = Y_train[i]
        
        difference = int(output_predicted[0]) - int(originalOutput)
        if(np.absolute(difference) == 0):
            num_correct = num_correct + 1;
    
accuracy_Training = num_correct /len(X_train)
print("Accuracy of Training DAta set.")
print(accuracy_Training);
    
num_correct = 0;
for i in range(0,len(X_validation)):
        output_predicted = clf.predict([X_validation[i]])
        originalOutput = Y_validation[i]
        
        difference = int(output_predicted[0]) - int(originalOutput)
        if(np.absolute(difference) == 0):
            num_correct = num_correct + 1;
    
accuracy_Validation = num_correct /len(X_validation)
print("Accuracy of Validation data set.")
print(accuracy_Validation);
    
    
    ########### Testing data set. ##########
with open('/MS_CS/Machine Learning/Assignment 1/ITCS6156_SLProject/DigitRecognition/optdigits_test.csv') as testingFile:
    testReader = csv.reader(testingFile)
    
    X_test = []
    Y_test = []
    
    for row in testReader:
        X_test.append(row[:64])
        Y_test.append(row[64])

        num_correct = 0;
        length = len(X_test)
        for i in range(0,length):
            output_predicted = clf.predict([X_test[i]])
            originalOutput = Y_test[i]
        
            difference = int(output_predicted[0]) - int(originalOutput)
            #print(difference)
            if(np.absolute(difference) == 0):
                num_correct = num_correct + 1;
    
accuracy = num_correct /len(X_test)
print("Test")
print(accuracy);