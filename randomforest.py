from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np

X = pd.read_csv("Data/data1.csv")

le = preprocessing.LabelEncoder()
le.fit(["NoEvents", "Attack", "Natural"])

y = le.transform(X['marker'])
X = X.drop(columns='marker')

X = X.replace(np.inf, np.finfo(np.float32).max)

clf = RandomForestClassifier()
clf.fit(X, y)

X1 = pd.read_csv("Data/data14.csv")

y1 = le.transform(X1['marker'])
X1 = X1.drop(columns='marker')
X1 = X1.replace(np.inf, np.finfo(np.float32).max)

y1p = clf.predict(X1)
print(np.sum(y1 == y1p)/y1.size)