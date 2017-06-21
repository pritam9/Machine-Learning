# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 18:51:10 2017

@author: Pritam
"""

import numpy as np
import pylab as pl
import pandas as pd
import re
from pandas import DataFrame
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV


train1 = pd.read_csv('/MS_CS/Machine Learning/Assignment 1/ReducedDataSet/amazonreduceddataset/amazon_baby_train_reduced.csv',header=1, names=['name', 'review', 'rating'])
train1.shape
train=train1.dropna()
train.shape
#test1 = pd.read_csv('/MS_CS/Machine Learning/Assignment 1/ITCS6156_SLProject/AmazonReviews/amazon_baby_test.csv',header=1, names=['name', 'review', 'rating'])
test1 = pd.read_csv('/MS_CS/Machine Learning/Assignment 1/ReducedDataSet/amazonreduceddataset/amazon_baby_test_reduced.csv',header=1, names=['name', 'review', 'rating'])
test1.shape
test = test1.dropna()
test.shape

print ('The first review is ')
print (train["review"][0])
example1 = BeautifulSoup(train["review"][0])
print (example1.get_text())
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
print (letters_only)
lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()
#nltk.download()  # Download text data sets, including stop words
print (stopwords.words("english")) 
words = [w for w in words if not w in stopwords.words("english")]
print (words)
input("Press Enter to continue...")
print ('Download text data sets. If you already have NLTK datasets downloaded, just close the Python download window...')
    #nltk.download()  # Download text data sets, including stop words

    # Initialize an empty list to hold the clean reviews
#clean_train_reviews = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list
print ("Cleaning and parsing the training set movie reviews...\n")

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    example2 = BeautifulSoup(raw_review)
    review_text = example2.get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 
# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in range( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    try:
        clean_train_reviews.append( review_to_words( train["review"][i] ) )
    except KeyError:
         continue
# Use skilearn to get values
print ("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()
# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print (vocab)
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print (count, tag)
print (test.shape)

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = [] 

print ("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    try:
        clean_review = review_to_words( test["review"][i] )
    except KeyError:
         continue
    clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
from sklearn.metrics import accuracy_score
model = SVC(kernel='linear', probability=False, C=10)
model=model.fit( train_data_features, train["rating"][:145035] )
print("Model is ready..")
result = model.predict(test_data_features)
#output = pd.DataFrame( data={"id":test["name"][:36208], "sentiment":result[:36208]} )
print("Accuracy Linear SVC c=10..")
#print(accuracy_score(output["sentiment"],test["rating"][:36208]))
print(accuracy_score(result,test["rating"]))

#Load Graph -- 
iris_dataset = load_iris()

#X, Y = iris_dataset.data, iris_dataset.target
X, Y = clean_train_reviews, clean_train_reviews
# It is usually a good idea to scale the data for SVM training.
# We are cheating a bit in this example in scaling all of the data,
# instead of fitting the transformation on the trainingset and
# just applying it on the test set.

scaler = StandardScaler()

X = scaler.fit_transform(X)

# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

C_range = 10. ** np.arange(-3, 8)
gamma_range = 10. ** np.arange(-5, 4)

param_grid = dict(gamma=gamma_range, C=C_range)

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(y=Y))

grid.fit(X, Y)

print("The best classifier is: ", grid.best_estimator_)

# plot the scores of the grid
# grid_scores_ contains parameter settings and scores
score_dict = grid.grid_scores_

# We extract just the scores
scores = [x[1] for x in score_dict]
scores = np.array(scores).reshape(len(C_range), len(gamma_range))

# Make a nice figure
pl.figure(figsize=(8, 6))
pl.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
pl.imshow(scores, interpolation='nearest', cmap=pl.cm.spectral)
pl.xlabel('gamma')
pl.ylabel('C')
pl.colorbar()
pl.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
pl.yticks(np.arange(len(C_range)), C_range)
pl.show()