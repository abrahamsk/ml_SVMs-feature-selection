#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Machine Learning 445
# HW 3: SVMs and Feature Selection
# Katie Abrahams, abrahake@pdx.edu
# 2/16/16

from exp1 import model, winner_index, winner
import heapq, random
import numpy as np

#####################################################################
# Experiment 2:
# Feature selection with linear SVM
# – Using final SVM model from Experiment 1:
#   1) • Obtain weight vector w
# Select features:
# – For m=1 to 57
#   2) • Select the set of m features that have highest |wm|
#   3) • Train a linear SVM, SVMm , on all the training data, only using
#        these m features, and using C* from Experiment 1
#   4) • Test SVMm on the test set to obtain accuracy.
# 5) – Plot accuracy vs. m
#####################################################################


# class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', ***coef0=0.0***,
# shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None,
# verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)[source]

# coef_ : array, shape = [n_class-1, n_features]
# Weights assigned to the features (coefficients in the primal problem).
# This is only available in the case of a linear kernel.
# coef_ is a readonly property derived from dual_coef_ and support_vectors_.

# 1) Obtain weight vector w from final SVM model from exp 1
# type(model._get_coef()) is numpy.ndarray
# w = model._get_coef()
# get absolute values for weights
weight_abs_values = np.absolute(model._get_coef())
# weight_abs_values.flatten()
# print weight_abs_values[0]
# print type(weight_abs_values)  # <type 'numpy.ndarray'>
# m_features = heapq.nlargest(5, range(len(weight_abs_values[0])), weight_abs_values.take)  # list
print weight_abs_values[0].max()

# get absolute values |w| and sort
# sorted_abs_list = sorted(map(abs, w))
# print type(sorted_abs_list)
# for n in sorted_abs_list:
#     sorted_abs_list.append(sorted(map(abs, n)))
    # print type(n) # <type 'numpy.ndarray'>
   # for m in n:
   #     print type(m) # <type 'numpy.float64'>


# For m = 1 to 57
# xrange is non-inclusive on upper bound, go to 58
for m in xrange(1, 58):
    # 2) Select the set of m features that have highest |wm|
    # get largest values for |wm|
    m_features = heapq.nlargest(m, weight_abs_values[0])
    print m_features
    # m_features = heapq.nlargest(m, w)
    # m_features = heapq.nlargest(m, enumerate(sorted_abs_list), key=lambda x:x[1])
    # print m_features

    # 3) Train a linear SVM, SVMm , on all the training data, only using
    # these m features, and using C* from Experiment 1

    # 4) Test SVMm on the test set to obtain accuracy.

# 5) – Plot accuracy vs. m
