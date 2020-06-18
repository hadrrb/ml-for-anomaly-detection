import pickle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

openfiles = ["output_Dataall.pickle"]
savefiles = ["results_scikitall.pdf"]

for openfile, savefile in zip(openfiles, savefiles):
    methods = ["RandomForest", "SVM", "MLP", "NaiveBayes"]
    all_acc, all_f1micro, all_f1macro, all_f1w, all_precisionmicro, all_precisionmacro, all_precisionw, all_recallmicro, all_recallmacro, all_recallw = pickle.load(open(openfile, "rb"))

    with PdfPages(savefile) as pdf:
        x = range(1,16)
        k = ["Dataset %d"%i for i in x]
        # plt.figure()
        # plt.plot(x, all_acc[0], '-go', label = methods[0])
        # plt.plot(x, all_acc[1], '-bo', label = methods[1])
        # plt.plot(x, all_acc[2], '-ro', label = methods[2])
        # plt.plot(x, all_acc[3], '-ko', label = methods[3])
        # plt.xticks(x, k, rotation=45)
        # plt.xlabel("Datasets")
        # plt.ylabel("Accuracy")
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
        # plt.ylim(0, 1)
        # pdf.savefig(bbox_inches='tight')

        # plt.close()
        # plt.clf()

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

