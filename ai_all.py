from sklearn import metrics, preprocessing

from sklearn.preprocessing import label_binarize
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
from mpi4py import MPI 
import sys
from scipy import interp
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sn
from itertools import cycle
import pickle

import matplotlib

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

methods = ["RandomForest", "SVM", "MLP", "NaiveBayes"]
methods = methods[:size]
#training

all_acc = []

all_f1micro = []
all_f1macro = []
all_f1w = []

all_precisionmicro = []
all_precisionmacro = []
all_precisionw = []

all_recallmicro = []
all_recallmacro = []
all_recallw = []

all_precision = []
all_recall = []
conf_matrix_list_of_arrays = []


yall = np.array([])
yall_score = []

X = []
for i in range(1,16):
    X.append(pd.read_csv(sys.argv[1] + "/data%d.csv"%i)) #read file

X = pd.concat(X)

X = X.replace(np.inf, np.finfo(np.float32).max) #replacing 'inf' with its equivalent in float32 datatype

#preparing the label converter

if sys.argv[1] != "multiclass":
    le = preprocessing.LabelEncoder()
    if sys.argv[1] == "Data":
        labels = ["Attack", "Natural", "NoEvents"]
    else:
        labels = ["Attack", "Natural"]
    le.fit(labels)
    y = le.transform(X['marker'])
else:
    y = X['marker'].values

#assigning the training data and the labels into variables

X = X.drop(columns='marker').values


clf = []
clf.insert(len(clf), RandomForestClassifier(n_estimators=100, max_features='log2')) #Random Forest classifier initialization
clf.insert(len(clf), SVC(probability=True, max_iter=1000, cache_size=7000))
clf.insert(len(clf), MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000, early_stopping=True))
clf.insert(len(clf), GaussianNB())

cv = StratifiedKFold(n_splits=10)

scores = cross_validate(clf[rank], X, y, cv=cv, scoring=['accuracy','f1_micro','f1_macro','f1_weighted','precision_micro','precision_macro','precision_weighted', 'recall_micro','recall_macro','recall_weighted'], n_jobs=2)
#GridSearchCV
y_pred = cross_val_predict(clf[rank], X, y, cv=cv, n_jobs=2)

all_acc.append(np.average(scores['test_accuracy']))

all_f1micro.append(np.average(scores["test_f1_micro"]))
all_f1macro.append(np.average(scores["test_f1_macro"]))
all_f1w.append(np.average(scores["test_f1_weighted"]))

all_precisionmicro.append(np.average(scores["test_precision_micro"]))
all_precisionmacro.append(np.average(scores["test_precision_macro"]))
all_precisionw.append(np.average(scores["test_precision_weighted"]))

all_recallmicro.append(np.average(scores["test_recall_micro"]))
all_recallmacro.append(np.average(scores["test_recall_macro"]))
all_recallw.append(np.average(scores["test_recall_weighted"]))


print("\n%s algorithm done!\n"%(methods[rank]))
sys.stdout.flush()
comm.Barrier()
all_acc = comm.gather(all_acc)

all_f1micro = comm.gather(np.average(all_f1micro))
all_f1macro = comm.gather(np.average(all_f1macro))
all_f1w = comm.gather(np.average(all_f1w))

all_precisionmicro = comm.gather(np.average(all_precisionmicro))
all_precisionmacro = comm.gather(np.average(all_precisionmacro))
all_precisionw = comm.gather(np.average(all_precisionw))

all_recallmicro = comm.gather(np.average(all_recallmicro))
all_recallmacro = comm.gather(np.average(all_recallmacro))
all_recallw = comm.gather(np.average(all_recallw))

if rank == 0:
  with open('output_' + sys.argv[1] +'all.pickle', 'wb') as results:
    pickle.dump([all_acc, all_f1micro, all_f1macro, all_f1w, all_precisionmicro, all_precisionmacro, all_precisionw, all_recallmicro, all_recallmacro, all_recallw], results)