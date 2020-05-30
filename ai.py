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
matplotlib.use('Agg')

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

labels = ["NoEvents", "Attack", "Natural"]
methods = ["RandomForest", "SVM", "MLP", "NaiveBayes", "NearestNeighbors"]
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

for i in range(1,16):
  print("\nDataset nr %d, %s algorithm\n"%(i, methods[rank]))
  sys.stdout.flush()
  X = pd.read_csv("Data/data%d.csv"%i) #read file

  X = X.replace(np.inf, np.finfo(np.float32).max) #replacing 'inf' with its equivalent in float32 datatype
  
  #preparing the label converter
  le = preprocessing.LabelEncoder()
  le.fit(labels)

  #assigning the training data and the labels into variables
  y = le.transform(X['marker'])
  X = X.drop(columns='marker').values


  clf = []
  clf.insert(len(clf), RandomForestClassifier(n_estimators=100, max_features='log2')) #Random Forest classifier initialization
  clf.insert(len(clf), SVC(probability=True, max_iter=1000, cache_size=7000))
  clf.insert(len(clf), MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000, early_stopping=True))
  clf.insert(len(clf), GaussianNB())

  cv = StratifiedKFold(n_splits=10)

  scores = cross_validate(clf[rank], X, y, cv=cv, scoring=['accuracy','f1_micro','f1_macro','f1_weighted','precision_micro','precision_macro','precision_weighted', 'recall_micro','recall_macro','recall_weighted'], n_jobs=2)
  #GridSearchCV
  y_pred = cross_val_predict(clf[rank], X, y, cv=cv, n_jobs=3)

  yall = np.insert(yall, len(yall), y)
  yall_score.insert(len(yall_score), cross_val_predict(clf[rank], X, y, cv=cv, method='predict_proba', n_jobs=3))
  
  conf_matrix_list_of_arrays.append(confusion_matrix(y, y_pred, labels=[0,1,2], normalize="all"))
 
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

yall_score = np.concatenate(yall_score, axis = 0)
y_bin = label_binarize(yall, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()
try:
  for k in range(n_classes):
      fpr[k], tpr[k], _ = roc_curve(y_bin[:, k], yall_score[:, k])
      roc_auc[k] = auc(fpr[k], tpr[k])
  colors = cycle(['blue', 'red', 'green'])
  for k, color in zip(range(n_classes), colors):
      plt.plot(fpr[k], tpr[k], color=color,
              label='ROC curve of class {0} (area = {1:0.2f})'
              ''.format(labels[k], roc_auc[k]))
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([-0.05, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic for %s classifier'%methods[rank])
  plt.legend(loc="lower right")
  pdf_roc = PdfPages("%s.pdf"%methods[rank])
  pdf_roc.savefig()
  plt.close()
  plt.clf()
except ValueError:
  pdf_roc = PdfPages("%s.pdf"%methods[rank])
  pass

mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)
df_cm = pd.DataFrame(mean_of_conf_matrix_arrays, index = labels, columns = labels)
f, ax = plt.subplots(figsize=(10,10))
sn.heatmap(df_cm, annot=True, square=True)
ax.set_ylim([0,3])
f.suptitle("Normalized confusion Matrix of " + methods[rank])
plt.xlabel("Predicted values")
plt.ylabel("True values")
pdf_roc.savefig()
plt.close()
plt.clf()
pdf_roc.close()
print(methods[rank], mean_of_conf_matrix_arrays, all_acc, all_f1macro, all_f1micro, all_f1w, all_precisionmacro, all_precisionmicro, all_precisionw, all_recallmacro, all_recallmicro, all_recallw)
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
  with open('output.pickle', 'wb') as results:
    pickle.dump([all_acc, all_f1micro, all_f1macro, all_f1w, all_precisionmicro, all_precisionmacro, all_precisionw, all_recallmicro, all_recallmacro, all_recallw], results)
  with PdfPages("results.pdf") as pdf:
    x = range(1,16)
    k = ["Dataset %d"%i for i in x]
    plt.figure()
    plt.plot(x, all_acc[0], '-bo', label = methods[0])
    plt.plot(x, all_acc[1], '-go', label = methods[1])
    plt.plot(x, all_acc[2], '-ro', label = methods[2])
    plt.plot(x, all_acc[3], '-yo', label = methods[3])
    plt.plot(x, all_acc[4], '-oo', label = methods[3])
    plt.xticks(x, k, rotation=45)
    plt.xlabel("Datasets")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.ylim(0, 1)
    pdf.savefig()

    plt.close()
    plt.clf()

    x = range(len(methods))
    plt.plot(x, all_f1micro, '-bo' , label = "micro")
    plt.plot(x, all_f1macro, '-go' , label = "macro")
    plt.plot(x, all_f1w, '-ro' , label = "weighted")
    plt.xlabel("Learners")
    plt.ylabel("F-measure")
    plt.xticks(x, methods)
    plt.ylim(0, 1)
    plt.legend(loc="best")
    pdf.savefig()

    plt.close()
    plt.clf()
 
    plt.plot(x, all_precisionmicro, '-bo' , label = "micro")
    plt.plot(x, all_precisionmacro, '-go' , label = "macro")
    plt.plot(x, all_precisionw, '-ro' , label = "weighted")
    plt.xticks(x, methods)
    plt.xlabel("Learners")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.ylim(0, 1)
    pdf.savefig()

    plt.close()
    plt.clf()

    plt.plot(x, all_recallmicro, '-bo' , label = "micro")
    plt.plot(x, all_recallmacro, '-go' , label = "macro")
    plt.plot(x, all_recallw, '-ro' , label = "weighted")
    plt.xlabel("Learners")
    plt.ylabel("Recall")
    plt.xticks(x, methods)
    plt.legend(loc="best")
    plt.ylim(0, 1)
    pdf.savefig()