import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split

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

clf = RandomForestClassifier(n_estimators=100, max_features='log2')

X=X.values

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf.fit(X_train,y_train)

explainer = LimeTabularExplainer(X_train, training_labels = y_train, feature_names = features, class_names = labels)

idx = 101

print('Event id: %d' % idx)
print('Predicted class =', labels[clf.predict(X_test[idx].reshape(1, -1))[0]])
print('True class: %s' % labels[y_test[idx]])

exp = explainer.explain_instance(X_test[idx], clf.predict_proba, num_features=128, labels=[0, 1, 2])

exp.save_to_file("lime.html")

from yellowbrick.features import rank1d, rank2d
from yellowbrick.model_selection.importances import feature_importances
from matplotlib import pyplot as plt

f, ax = plt.subplots(ncols=2)

rank1d(X, y, ax[0], features= features, show=False)
rank2d(X, y, ax[1], features= features, show=False)

ax[0].tick_params(axis='both', which='major', labelsize=2)
ax[0].tick_params(axis='both', which='minor', labelsize=2)
ax[1].tick_params(axis='both', which='major', labelsize=1)
ax[1].tick_params(axis='both', which='minor', labelsize=1)

plt.savefig("feat_ranking_yellowbrick.pdf", dpi=2000)

f2, ax2 = plt.subplots()
feature_importances(clf, X, y, ax2, labels = features, show=False)
ax2.tick_params(axis='both', which='major', labelsize=1)
ax2.tick_params(axis='both', which='minor', labelsize=1)

plt.savefig("feat_importance_yellowbrick.pdf", dpi=2000)

import eli5
from eli5.sklearn import PermutationImportance
from eli5.ipython import show_weights
perm = PermutationImportance(clf).fit(X_test, y_test)
with open('feat_importance_eli5.html', 'w') as f:
    f.write(show_weights(perm, target_names = labels, feature_names = features, top = None, show = ['feature_importances', 'targets', 'method', 'description']).data)