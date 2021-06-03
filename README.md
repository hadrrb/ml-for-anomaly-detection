# <i>Master Thesis:</i> Hybrid neural networks for anomaly detection in cyber-physical systems

This repository contains the source code used to produce the results for the master thesis (in Python3) in the main directory and the source code of the master thesis itself (in LaTeX) in <a href="thesis">thesis folder</a>.

## Abstract

Nowadays cyber-physical systems are widely used in different application domains. In parallel, machine learning algorithms are used widely to detect the anomalies in the behaviour of these systems. However, this detection is limited to two states: normal behaviour and faulty functioning. This master thesis aims to extend this detection to differentiate between attacks and normal faults. In first place, a power system is described as an example to work on. Then, various machine learning algorithms are evaluated on the given datasets, and this using two machine learning toolkits - scikit-learn and Weka. Later, various tools for feature analysis are presented and an algorithm to find the features that contributed the most into the false predictions is described. Finally, three solutions to the initial problem are presented and evaluated.


## Repository content description

The integral text of the master thesis can be found in <a href="thesis/thesis.pdf">this pdf file</a>. Below is presented the source code used for each of the chapters in the thesis.

### Chapter 2: Power system as a CPS example

- <a href="files_calc.ipynb">files_calc.ipnyb</a>: conversion of dataset from `.arff` to `.csv` and analysis of distribution of classes throughout files.

### Chapter 3: Machine learning algorithms comparison

- <a href="ai_all.py">ai_all.py</a>: script to calculate comparison metrics values for all the classifiers for the 3 available datasets (multiclass, binary, three classes). As output il creates `pickle` files containing the results to be processed afterwards,
- <a href="plot.ipynb">plot.ipynb</a>: tool for creating plots for all comparison metrics using `pickle` files created by the previous script,
- <a href="roc.py">roc.py</a>: a script to create roc curves for classifiers running on binary data (not displayed in the thesis),
- <a href="ai.py">ai.py</a>: legacy script for calculate comparison metrics values for all the classifiers for 3 class dataset. It creates also the ROC curve and the confusion matrix,
- <a href="ai_binary.py">ai_binary.py</a>: legacy script for calculate comparison metrics values for all the classifiers for binary dataset,
- <a href="ai_multiclass.py">ai_multiclass.py</a>: legacy script for calculate comparison metrics values for all the classifiers for multiclass dataset,
- <a href="proc.py">proc.py</a>: script for converting `csv` to `arff` in order to run tests in WEKA,
- <a href="plotting.py">plotting.py</a>: legacy script for creating plots for all comparison metrics,
- <a href="param_optim.ipynb">param_optim.ipynb</a>: script for finding the best set of parameters for the discussed classifiers.

### Chapter 4: Features' importance

- <a href="features.ipynb">features.ipynb</a>: checking the capabilities of LIME and YellowBrick packages,
- <a href="trees_visualisation.ipynb">trees_visualisation.ipynb</a>: checking the capabilities of dtreeviz,
- <a href="lime_features_classification.py">lime_features_classification.py</a>: getting the features' importances using LIME package (algorithm discussed on page 53 of the thesis.

### Chapter 5: Model enhancement

- <a href="featfun.ipynb">featfun.ipynb</a>: attempt to enhance the predictions of Decision Tree classifier,
- <a href="featfun_rf.ipynb">featfun_rf.ipynb</a>: attempt to enhance the predictions of Random Forest classifier,
- <a href="featfun_mlp.ipynb">featfun_mlp.ipynb</a>: attempt to enhance the predictions of Multilayer Perceptron classifier,
