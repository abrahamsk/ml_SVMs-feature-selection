#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Machine Learning 445
# HW 3: SVMs and Feature Selection
# Katie Abrahams, abrahake@pdx.edu
# 2/16/16

from input import *
from sklearn.cross_validation import KFold

###########################################################################
# Experiment 1:
# Cross-validation using linear SVM to find best "C" parameter value.
# - Use linear kernel, Use 10-fold cross-validation to test values of C:
# {0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1}
# 1)  Split training set into 10 approx equal-sized disjoint sets Si
# 2)  For each value of the C parameter j:
#     – For i=1 to 10:
#           Select Si to be the "validation" set
#           Train linear SVM using C=j and all training data except Si.
#           Test learned model on Si to get accuracy Aj,i
# 3)    – Compute average accuracy for C=j: Aj = Σ(i=1,10) Aj,i
# 4)  Choose value C= C* that results in highest Aj
# 5) Train new linear SVM using all training data with C=C*
# 6) - Test learned SVM model on test data. Report accuracy, precision,
#   and recall (using threshold 0 to determine pos and neg classifications)
# 7) – Use results on test data to create an ROC curve for this SVM,
#   using about 200 evenly spaced thresholds.
###########################################################################

# 1) Split training data
# print X_scaled.shape  # 1810x57
# print X_concat.shape  # 1810x58

# 1810 rows/10 = 181 rows per disjoint set
# sklearn.cross_validation.KFold(n, n_folds=3, shuffle=False, random_state=None)
# n: int Total number of elements
# n_folds: int, Number of folds
# The first n % n_folds folds have size n // n_folds + 1, other folds have size n // n_folds.
kf = KFold(1810, n_folds=10)
# print len(kf)  # 10


