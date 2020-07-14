import pandas as pd
import numpy as np
from sklearn import preprocessing
import arff

labels = ["NoEvents", "Natural", "Attack"]
for i in range(1,16):
  X = pd.read_csv("Data/data%d.csv"%i) #read file

  X = X.replace(np.inf, np.finfo(np.float32).max) #replacing 'inf' with its equivalent in float32 datatype
  
  #preparing the label converter
  le = preprocessing.LabelEncoder()
  le.fit(labels)

  #assigning the training data and the labels into variables
  X['marker'] = le.transform(X['marker'])

  arff.dump('Data_out/data%d.arff'%i
      , X.values
      , relation='relation name'
      , names=X.columns)