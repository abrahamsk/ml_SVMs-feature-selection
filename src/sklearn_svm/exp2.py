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
from sklearn import svm, metrics
import matplotlib.pyplot as plt

####################################################################
# Experiment 2:
# Feature selection with linear SVM
# – Using final SVM model from Experiment 1:
#   1) • Obtain weight vector w
# Select features:
# – For m=1 to 57
#   2) • Select the set of m features that have highest |wm|
#   3) • Train a linear SVM, SVMm , on all the training data,
#        only using these m features, and using C* from Experiment 1
#   4) • Test SVMm on the test set to obtain accuracy.
# 5) – Plot accuracy vs. m
####################################################################

def print_stats():
    """
    Print stats about experiment 2 data
    :return nothing, but print stats:
    """
    print "----- Exp 2 Accuracy -----\n",acc,"\nC =",winner,"\nNum features =",m,"\n--------------------------"

    # Performance metrics
    # Returns text summary of the precision, recall, F1 score for each class
    target_names = ['Not spam', 'Spam']
    print "Classification report for %s" % svm_m
    print metrics.classification_report(X_test_col, predicted, target_names=target_names)

############################################################################################


# class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', ***coef0=0.0***,
# shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None,
# verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)[source]

# coef_ : array, shape = [n_class-1, n_features]
# Weights assigned to the features (coefficients in the primal problem).
# This is only available in the case of a linear kernel.
# coef_ is a readonly property derived from dual_coef_ and support_vectors_.

# 1) Obtain weight vector w from final SVM model from exp 1
# get absolute values for weights
weight_abs_values = np.absolute(model._get_coef())

# list to store accuracies of svm running on test data
# using m features subset
acc_list = []
# list to save which indices (features) were selected
features_selected = []

# For m = 2 to 57 (need to have at least 2 features for the SVM to work)
# xrange is non-inclusive on upper bound, go to 58
for m in xrange(2, 58):
    # 2) Select the set of m features that have highest |wm|
    # get m largest values for |wm|

    # get indices of m largest weight values
    indices = heapq.nlargest(m, range(len(weight_abs_values[0])), weight_abs_values[0].take)

    # save which indices (features) were selected
    features_selected.append(indices)

    # get features corresponding to largest weights
    X_scaled_subset = np.copy(X_scaled[:,[i for i in indices]])

    # 3) Train a linear SVM, SVMm , on all the training data, only using
    # these m features, and using C* from Experiment 1
    svm_m = svm.SVC(C=winner, kernel='linear')
    # Fit the SVM model according to the given training data subset (m features)
    svm_m.fit(X_scaled_subset, X_col.ravel())

    # 4) Test SVMm on the test set to obtain accuracy.
    # must test using the same number of features as training
    acc = svm_m.score(X_test_scaled[:,[i for i in indices]], X_test_col)
    acc_list.append(acc)
    # Predict class labels for samples in X_test_scaled.
    predicted = svm_m.predict(X_test_scaled[:,[i for i in indices]])
    print_stats()

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
    plt.title('Accuracy vs. m features')
    plt.legend(loc="lower right")
    plt.show()

############################################################################################

def get_top_features():
    """
    Get top 5 features based on accuracy of SVM
    Using spambase.names to identify features
    :return:
    """

    # features from spambase.names
    feature_vals = ["word_freq_make",
                    "word_freq_address",
                    "word_freq_all",
                    "word_freq_3d",
                    "word_freq_our",
                    "word_freq_over",
                    "word_freq_remove",
                    "word_freq_internet",
                    "word_freq_order",
                    "word_freq_mail",
                    "word_freq_receive",
                    "word_freq_will",
                    "word_freq_people",
                    "word_freq_report",
                    "word_freq_addresses",
                    "word_freq_free",
                    "word_freq_business",
                    "word_freq_email",
                    "word_freq_you",
                    "word_freq_credit",
                    "word_freq_your",
                    "word_freq_font",
                    "word_freq_000",
                    "word_freq_money",
                    "word_freq_hp",
                    "word_freq_hpl",
                    "word_freq_george",
                    "word_freq_650",
                    "word_freq_lab",
                    "word_freq_labs",
                    "word_freq_telnet",
                    "word_freq_857",
                    "word_freq_data",
                    "word_freq_415",
                    "word_freq_85",
                    "word_freq_technology",
                    "word_freq_1999",
                    "word_freq_parts",
                    "word_freq_pm",
                    "word_freq_direct",
                    "word_freq_cs",
                    "word_freq_meeting",
                    "word_freq_original",
                    "word_freq_project",
                    "word_freq_re",
                    "word_freq_edu",
                    "word_freq_table",
                    "word_freq_conference",
                    "char_freq_;",
                    "char_freq_(",
                    "char_freq_[",
                    "char_freq_!",
                    "char_freq_$",
                    "char_freq_# ",
                    "capital_run_length_average",
                    "capital_run_length_longest",
                    "capital_run_length_total"]

    # get the highest value and index of that accuracy from accuracy list
    max_val = max(acc_list)
    max_index = acc_list.index(max_val)

    # get features selected when accuracy of SVM was highest
    # print "Features selected with highest accuracy:",features_selected[max_index]
    # store features selected to display feature values
    print "Feature values for features with highest accuracy:"
    for i in features_selected[max_index]:
        print feature_vals[i]

############################################################################################

# Only plot accuracy and fetch highest-accuracy features if exp2 is the main program
if __name__ == "__main__":
  plot()
  get_top_features()


