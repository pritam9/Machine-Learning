# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 19:09:16 2017

@author: Pritam
"""

import pandas as pd
import numpy as np
import nltk
import string
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.tree import DecisionTreeClassifier


# In[ ]:

reviews = pd.read_csv('/MS_CS/Machine Learning/Assignment 1/ITCS6156_SLProject/AmazonReviews/amazon_baby_train.csv')
reviews.shape
reviews = reviews.dropna()
reviews.shape
#print(reviews.head(25))

scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: 'pos' if x > 3 else 'neg')
#print(reviews.head(25))


print(scores.mean())
print(scores.std())


# In[ ]:

reviews.groupby('rating')['review'].count()


# In[ ]:

reviews.groupby('rating')['review'].count().plot(kind='bar', color= ['r','g'],title='Label Distribution',  figsize = (10,6))


# In[ ]:

def splitPosNeg(Summaries):
    neg = reviews.loc[Summaries['rating']== 'neg']
    pos = reviews.loc[Summaries['rating']== 'pos']
    return [pos,neg]
    


# In[ ]:

[pos,neg] = splitPosNeg(reviews)


# In[ ]:

#preprocessing steps
#nltk.download()
#stemmer = PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()
#stop = stopwords.words('english')
translation = str.maketrans(string.punctuation,' '*len(string.punctuation))

#filtered_words = [word for word in word_list if word not in stopwords.words('english')]

def preprocessing(line):
    tokens=[]
    line = line.translate(translation)
    line = nltk.word_tokenize(line.lower())
    #print(line)
    #stops = stopwords.words('english')
    #stops.remove('not')
    #stops.remove('no')
    #line = [word for word in line if word not in stops]
    #print("After removing stop words")
    #print(line)
    for t in line:
        #if(t not in stop):
            #stemmed = stemmer.stem(t)
        stemmed = lemmatizer.lemmatize(t)
        tokens.append(stemmed)
    return ' '.join(tokens)


# In[ ]:

pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done")


# In[ ]:

data = pos_data + neg_data
labels = np.concatenate((pos['rating'].values,neg['rating'].values))
#print(labels)


# In[ ]:

[Data_train,Data_test,Train_labels,Test_labels] = train_test_split(data,labels , test_size=0.25, random_state=20160121,stratify=labels)


# In[ ]:

t = []
for line in data:
    l = nltk.word_tokenize(line)
    for w in l:
        t.append(w)
#print(t)


# In[ ]:

word_features = nltk.FreqDist(t)
print(len(word_features))


# In[ ]:

topwords = [fpair[0] for fpair in list(word_features.most_common(5000))]
print(word_features.most_common(25))


# In[ ]:

word_his = pd.DataFrame(word_features.most_common(200), columns = ['words','count'])
#print(word_his)


# In[ ]:

vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])
print(c_fit)


# In[ ]:

tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)


# In[ ]:

ctr_features = vec.transform(data)
tr_features = tf_vec.transform(ctr_features)


# In[ ]:

tr_features.shape


# In[ ]:

#cte_features = vec.transform(Data_test)
#te_features = tf_vec.transform(cte_features)


# In[ ]:

clf = DecisionTreeClassifier()
clf = clf.fit(tr_features, labels)

tfPredication = clf.predict(tr_features)
tfAccuracy = metrics.accuracy_score(tfPredication,labels)
print(tfAccuracy)


# In[ ]:

print(metrics.classification_report(labels, tfPredication))


# In[ ]:

#Testing set


# In[ ]:

reviews = pd.read_csv('/MS_CS/Machine Learning/Assignment 1/ITCS6156_SLProject/AmazonReviews/amazon_baby_test.csv')
reviews.shape
reviews = reviews.dropna()
reviews.shape
#print(reviews.head(25))

scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: 'pos' if x > 3 else 'neg')
#print(reviews.head(25))


scores.mean()


# In[ ]:

reviews.groupby('rating')['review'].count()


# In[ ]:

reviews.groupby('rating')['review'].count().plot(kind='bar', color= ['r','g'],title='Label Distribution',  figsize = (10,6))


# In[ ]:

[pos,neg] = splitPosNeg(reviews)


# In[ ]:

pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done")


# In[ ]:

data = pos_data + neg_data
labels = np.concatenate((pos['rating'].values,neg['rating'].values))
#print(labels)


# In[ ]:

t = []
for line in data:
    l = nltk.word_tokenize(line)
    for w in l:
        t.append(w)
#print(t)


# In[ ]:

word_features = nltk.FreqDist(t)
print(len(word_features))


# In[ ]:

topwords = [fpair[0] for fpair in list(word_features.most_common(5002))]
print(word_features.most_common(25))


# In[ ]:

word_his = pd.DataFrame(word_features.most_common(200), columns = ['words','count'])
#print(word_his)


# In[ ]:

len(topwords)


# In[ ]:

vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])


# In[ ]:

tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)


# In[ ]:

cte_features = vec.transform(data)
te_features = tf_vec.transform(cte_features)


# In[ ]:

te_features.shape


# In[ ]:

tePredication = clf.predict(te_features)
teAccuracy = metrics.accuracy_score(tePredication,labels)
print(teAccuracy)


# In[ ]:

print(metrics.classification_report(labels, tePredication))


# In[ ]: