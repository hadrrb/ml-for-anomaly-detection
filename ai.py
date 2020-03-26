from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, preprocessing
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
from mpi4py import MPI 
import sys

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

labels = ["NoEvents", "Attack", "Natural"]
#training
X = []
for i in range(1,16):
    X.insert(len(X), pd.read_csv("Data/data%d.csv"%i)) #read file

X = pd.concat(X)
X = X.replace(np.inf, np.finfo(np.float32).max) #replacing 'inf' with its equivalent in float32 datatype

X1 = X.loc[X['marker'] == labels [0]]
X2 = X.loc[X['marker'] == labels [1]]
X3 = X.loc[X['marker'] == labels [2]]

def rand(X):
  nbr = X.shape[0]
  ele = np.arange(nbr)
  rng = np.random.default_rng()
  randpos = rng.choice(ele, size = int(nbr*0.9), replace = False)
  Xtrain = X.iloc[randpos]
  Xpred = X.iloc[ele[[k not in randpos for k in ele]]]
  return Xtrain, Xpred

X1train, X1pred = rand(X1)
X2train, X2pred = rand(X2)
X3train, X3pred = rand(X3)

Xtrain = pd.concat([X1train, X2train, X3train])
Xpred = pd.concat([X1pred, X2pred, X3pred])

#preparing the label converter
le = preprocessing.LabelEncoder()
le.fit(labels)

#assigning the training data and the labels into variables
ytrain = le.transform(Xtrain['marker'])
Xtrain = Xtrain.drop(columns='marker') 

y_test = le.transform(Xpred['marker'])
X_test = Xpred.drop(columns='marker') 


clf = []
clf.insert(len(clf), RandomForestClassifier()) #Random Forest classifier initialization
clf.insert(len(clf), SVC(max_iter=100))
clf.insert(len(clf), MLPClassifier())


clf[rank].fit(Xtrain, ytrain) #teaching the dataset
predicted = clf[rank].predict(X_test) #predicting the labels
print("Classification report for classifier %s:\n%s\n"
% (clf[rank], metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(clf[rank], X_test, y_test, display_labels= labels)
print("Confusion matrix:\n%s" % disp.confusion_matrix)
disp.figure_.suptitle("Confusion Matrix of %s"%type(clf[rank]).__name__)
sys.stdout.flush()
plt.title = type(clf[rank]).__name__
plt.show()
    
