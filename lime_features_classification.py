import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pickle
from lime.lime_tabular import LimeTabularExplainer

labels = ["NoEvents", "Attack", "Natural"]

X = pd.read_csv("Data/data%d.csv"%1) #read file

X = X.replace(np.inf, np.finfo(np.float32).max) #replacing 'inf' with its equivalent in float32 datatype

#preparing the label converter
le = preprocessing.LabelEncoder()
le.fit(labels)

#assigning the training data and the labels into variables
y = le.transform(X['marker'])
X = X.drop(columns='marker')

features = list(X.columns)

classifiers = [DecisionTreeClassifier(), RandomForestClassifier(n_estimators=100, max_features='log2'), MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000, early_stopping=True)]
clftxt = ["DecisionTree", "RandomForest" , "MLP"]
X=X.values

for clf, filen in zip(classifiers, clftxt):
    clf.fit(X,y)

    explainer = LimeTabularExplainer(X, training_labels = y, feature_names = features, class_names = labels)


    Xall = []
    for i in range(2,16):
        Xall.append(pd.read_csv("Data/data%d.csv"%i))

    Xall = pd.concat(Xall)
    Xall = Xall.replace(np.inf, np.finfo(np.float32).max) #replacing 'inf' with its equivalent in float32 datatype

    #assigning the training data and the labels into variables
    yall = le.transform(Xall['marker'])
    Xall = Xall.drop(columns='marker').values

    boole = (yall != clf.predict(Xall))
    res = []
    for label in range(0,3):
        faulty = boole & (yall == label)
        X_test = Xall[faulty]
        y_test = yall[faulty]
        lst = []
        for idx in range(0, 100):
            exp = explainer.explain_instance(X_test[idx], clf.predict_proba, num_features=128, labels=[0, 1, 2])
            lst.append(exp.as_list(label=label))
        lst = np.array(lst)
        clst = np.concatenate(lst, axis=0)
        dtfr = pd.DataFrame(clst, columns=['feature', 'importance'])
        dtfr["importance"] = pd.to_numeric(dtfr["importance"])
        dtfr = dtfr.groupby(['feature']).mean()
        dtfr= dtfr.sort_values(by="importance")
        res.append(dtfr)

    with open(filen+'.lime', 'wb') as lime:
        pickle.dump(res, lime)