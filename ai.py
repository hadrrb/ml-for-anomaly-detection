from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, preprocessing
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
from mpi4py import MPI 
import sys
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc, plot_roc_curve, roc_curve
from sklearn.model_selection import StratifiedKFold
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.naive_bayes import GaussianNB

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

labels = ["NoEvents", "Attack", "Natural"]
methods = ["RandomForest", "SVM", "MLP", "NaiveBayes"]
#training

all_acc = []
all_f1 = []
all_precision = []
all_recall = []

for i in range(1,16):
  X = pd.read_csv("Data/data%d.csv"%i) #read file

  X = X.replace(np.inf, np.finfo(np.float32).max) #replacing 'inf' with its equivalent in float32 datatype
  
  #preparing the label converter
  le = preprocessing.LabelEncoder()
  le.fit(labels)

  #assigning the training data and the labels into variables
  y = le.transform(X['marker'])
  X = X.drop(columns='marker').values
  

  clf = []
  clf.insert(len(clf), RandomForestClassifier()) #Random Forest classifier initialization
  clf.insert(len(clf), SVC())
  clf.insert(len(clf), MLPClassifier(hidden_layer_sizes=(20,)))
  clf.insert(len(clf), GaussianNB())
  #adaboost

  cv = StratifiedKFold(n_splits=10)

  tprs = []
  aucs = []
  acc = []
  f1 = []
  precision = []
  recall = []
  mean_fpr = np.linspace(0, 1, 100)

  # fig, ax = plt.subplots()

  with PdfPages(methods[rank] + '/%d.pdf'%i) as pdf:
    for k, (train, test) in enumerate(cv.split(X, y)):
        clf[rank].fit(X[train], y[train])
        # viz = plot_roc_curve(clf[rank], X[test], y[test],
        #                     name='ROC fold {}'.format(i),
        #                     alpha=0.3, lw=1, ax=ax)
        txt = metrics.classification_report(y[test], clf[rank].predict(X[test]))
        res = metrics.classification_report(y[test], clf[rank].predict(X[test]), output_dict=True)
        # interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        # interp_tpr[0] = 0.0
        # tprs.append(interp_tpr)
        # aucs.append(viz.roc_auc)
        acc.append(res['accuracy'])
        f1.append(res['macro avg']['f1-score'])
        precision.append(res['macro avg']['precision'])
        recall.append(res['macro avg']['recall'])
        disp = metrics.plot_confusion_matrix(clf[rank], X[test], y[test], display_labels= labels)
        disp.figure_.suptitle("Confusion Matrix of the %d fold"%k)
        pdf.savefig()
        plt.close()
        f = plt.figure()
        plt.clf()
        plt.axis('off')
        np.set_printoptions(suppress=True)
        plt.text(0.5,0.5, txt, transform=f.transFigure, size=10, ha="center", wrap = True)
        pdf.savefig(f)
        plt.close()

    # ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    #         label='Chance', alpha=.8)

    # mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    # ax.plot(mean_fpr, mean_tpr, color='b',
    #         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
    #         lw=2, alpha=.8)

    # std_tpr = np.std(tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                 label=r'$\pm$ 1 std. dev.')

    # ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
    #       title="Receiver operating characteristic example")
    # ax.legend(loc="lower right")
    # pdf.savefig(fig)

  
  acc = np.average(acc)
  f1 = np.average(f1)
  precision = np.average(precision)
  recall = np.average(recall)

  #print("Dataset %d, acc = %.3f, f1 = %.3f, precision = %.3f, recall = %.3f"%(i, acc, f1, precision, recall))

  all_acc.append(acc)
  all_f1.append(f1)
  all_precision.append(precision)
  all_recall.append(recall)


comm.Barrier()
all_acc = comm.gather(all_acc)
all_f1 = comm.gather(np.average(all_f1))
all_precision = comm.gather(np.average(all_precision))
all_recall = comm.gather(np.average(all_recall))

if rank == 0:
  with PdfPages("results.pdf") as pdf:
    x = range(1,16)
    k = ["Dataset %d"%i for i in y]
    plt.plot(x, all_acc[0], '-bo', label = methods[0])
    plt.plot(x, all_acc[1], '-go', label = methods[1])
    plt.plot(x, all_acc[2], '-ro', label = methods[2])
    plt.plot(x, all_acc[3], '-yo', label = methods[3])
    plt.xticks(x, k)
    plt.xlabel("Datasets")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right")
    pdf.savefig()

    plt.close()
    plt.clf()

    x = range(len(methods))
    plt.plot(x, all_f1, '-bo')
    plt.xlabel("Learners")
    plt.ylabel("F-measure")
    plt.xticks(x, methods)

    pdf.savefig()

    plt.close()
    plt.clf()
 
    plt.plot(x, all_precision, '-bo')
    plt.xticks(x, methods)
    plt.xlabel("Learners")
    plt.ylabel("Precision")

    pdf.savefig()

    plt.close()
    plt.clf()

    plt.plot(x, all_recall, '-bo')
    plt.xlabel("Learners")
    plt.ylabel("Recall")
    plt.xticks(x, methods)

    pdf.savefig()







    
