from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, preprocessing
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from matplotlib import pyplot as plt

#training
X = []
for i in range(1,16):
    X.insert(len(X), pd.read_csv("Data/data%d.csv"%i)) #read file

X = pd.concat(X)

X = X.replace(np.inf, np.finfo(np.float32).max) #replacing 'inf' with its equivalent in float32 datatype

nbr = X.shape[0]
ele = np.arange(nbr)
rng = np.random.default_rng()
randpos = rng.choice(ele, size = int(nbr*0.3), replace = False)

Xtrain = X.iloc[randpos]
Xpred = X.iloc[ele[[k not in randpos for k in ele]]]

#preparing the label converter
le = preprocessing.LabelEncoder()
le.fit(["NoEvents", "Attack", "Natural"])

#assigning the training data and the labels into variables
ytrain = le.transform(Xtrain['marker'])
Xtrain = Xtrain.drop(columns='marker') 

y_test = le.transform(Xpred['marker'])
X_test = Xpred.drop(columns='marker') 


clf = []
clf.insert(len(clf), RandomForestClassifier()) #Random Forest classifier initialization
clf.insert(len(clf), SVC())
clf.insert(len(clf), Perceptron())
SVC().__str__
for k in clf:
    k.fit(Xtrain, ytrain) #teaching the dataset
    predicted = k.predict(X_test) #predicting the labels
    print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(y_test, predicted)))
    disp = metrics.plot_confusion_matrix(k, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix of %s"%k.__str__)
    print("Confusion matrix:\n%s" % disp.confusion_matrix)

plt.show()
    
