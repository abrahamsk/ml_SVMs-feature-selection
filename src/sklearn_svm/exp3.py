#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Machine Learning 445
# HW 3: SVMs and Feature Selection
# Katie Abrahams, abrahake@pdx.edu
# 2/16/16

from exp1 import model, winner
from input import *
import heapq
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

##########################################################################################
# Experiment 3: Random feature selection
# Same as Experiment 2, but for each m, select m features at random from the complete set.
# This is to see if using SVM weights for feature selection has any advantage over random.
#
# Exp3 repeats much of exp2's code, with modifications for new feature selection rules.
##########################################################################################

# 1) Obtain weight vector w from final SVM model from exp 1
# get absolute values for weights
weight_abs_values = np.absolute(model._get_coef())

# list to store accuracies of svm running on test data
# using m features subset
acc_list = []
# For m = 2 to 57 (need to have at least 2 features for the SVM to work)
for m in xrange(2, 58):
    # 2) Select the set of m features that have highest |wm|
    # get m largest values for |wm|

    # select m features at random from the complete set (without replacement)
    indices = np.random.choice(range(len(weight_abs_values[0])), m, replace=False)
    # get features corresponding to selected weights
    X_scaled_subset = np.copy(X_scaled[:,[i for i in indices]])

    # 3) Train a linear SVM, SVMm , on all the training data, only using
    # these m features, and using C* from Experiment 1
    svm_m = svm.SVC(C=winner, kernel='linear')
    # Fit the SVM model according to the given training data subset (m features)
    svm_m.fit(X_scaled_subset, X_col.ravel())

    # 4) Test SVMm on the test set to obtain accuracy.
    # must test using the same number of features as training
    acc_list.append(svm_m.score(X_test_scaled[:,[i for i in indices]], X_test_col))
    # Predict class labels for samples in X_test_scaled.
    predicted = svm_m.predict(X_test_scaled[:,[i for i in indices]])

# 5) – Plot accuracy vs. m
def plot():
    """
    Plot accuracy of SVM on test data vs. m number of features
    :return:
    """
    plt.plot(range(2, 58), acc_list, label='Accuracy')
    plt.xlim([1.0, 59.0])
    plt.ylim([0.0, 1.05])
    # label x and y axes, generate title
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy of SVM (Test data)')
    plt.title('Accuracy vs. m features, randomly selected')
    plt.legend(loc="lower right")
    plt.show()

# Only plot accuracy if exp3 is the main program
if __name__ == "__main__":
  plot()