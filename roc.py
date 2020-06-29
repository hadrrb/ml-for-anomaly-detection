import pandas as pd
import numpy as np
from sklearn import preprocessing
from yellowbrick.classifier.rocauc import roc_auc
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

labels = ["NoEvents", "Attack", "Natural"]

X = []
for i in range(1,16):
    X.append(pd.read_csv("C:/Users/bhadr/Documents/Programming/Stage2020/binary/data%d.csv"%i)) #read file

X = pd.concat(X)

X = X.replace(np.inf, np.finfo(np.float32).max)

le = preprocessing.LabelEncoder()
le.fit(labels)
y = le.transform(X['marker'])
#y = X['marker'].values
X = X.drop(columns='marker').values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

methods = [RandomForestClassifier(n_estimators=100, max_features='log2'), SVC(probability=True),  MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000, early_stopping=True), GaussianNB()]

with PdfPages("roc_2c.pdf") as pdf:
    for method in methods:
        f, ax = plt.subplots()
        roc_auc(method, X_train, y_train, X_test=X_test, y_test=y_test, ax= ax,classes = labels)
        #ax.get_legend().remove()
        pdf.savefig(f)
        plt.close()
        plt.clf()


