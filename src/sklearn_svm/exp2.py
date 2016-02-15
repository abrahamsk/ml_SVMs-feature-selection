#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Machine Learning 445
# HW 3: SVMs and Feature Selection
# Katie Abrahams, abrahake@pdx.edu
# 2/16/16

from exp1 import model

#####################################################################
# Experiment 2:
# Feature selection with linear SVM
# – Using final SVM model from Experiment 1:
#   • Obtain weight vector w
# Select features:
# – For m=1 to 57
#   • Select the set of m features that have highest |wm|
#   • Train a linear SVM, SVMm , on all the training data, only using
#       these m features, and using C* from Experiment 1
#   • Test SVMm on the test set to obtain accuracy.
# – Plot accuracy vs. m
#####################################################################


# class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
# shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None,
# verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)[source]
print model._get_coef()
# coef_ : array, shape = [n_class-1, n_features]
# Weights assigned to the features (coefficients in the primal problem).
# This is only available in the case of a linear kernel.
# coef_ is a readonly property derived from dual_coef_ and support_vectors_.
