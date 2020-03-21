from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron

#training
X = pd.read_csv("Data/data1.csv") #read file

#preparing the label converter
le = preprocessing.LabelEncoder()
le.fit(["NoEvents", "Attack", "Natural"])

#assigning the training data and the labels into variables
y = le.transform(X['marker'])
X = X.drop(columns='marker')

X = X.replace(np.inf, np.finfo(np.float32).max) #replacing 'inf' with its equivalent in float32 datatype

clf = []
clf.insert(len(clf), RandomForestClassifier()) #Random Forest classifier initialization
clf.insert(len(clf), SVC())
clf.insert(len(clf), Perceptron())

for k in clf:
    k.fit(X, y) #teaching the dataset

def predict(clf, le, file): #prediction function based on the prepared classifier
    X1 = pd.read_csv(file) #reading file

    y1 = le.transform(X1['marker']) #saving the labels after conversion into numerical values
    X1 = X1.drop(columns='marker') #saving the data
    X1 = X1.replace(np.inf, np.finfo(np.float32).max) #replacing 'inf' with its equivalent in float32 datatype

    y1p = clf.predict(X1) #predicting the labels
    return np.sum(y1 == y1p)/y1.size #calculation of rate of success of the algorithm for the given dataset

for i in range(2,16): #iterations over all available datasets
    out = []
    for k in clf:
        out.insert(len(out), predict(k, le, "Data/data" + str(i) +".csv"))
    print("Dataset nr %d was recognized with a success rate of %2.2f %s (Random Forests), %2.2f %s (SVM), %2.2f %s (MLP)." % (i, (out[0] * 100), '%', (out[1] * 100), '%' , (out[2] * 100), '%'))