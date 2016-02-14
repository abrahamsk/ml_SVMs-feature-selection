#!/usr/bin/env python
# -*- coding: utf-8 -*-
##  utf-8 for non-ASCII chars

# Machine Learning 445
# HW 3: SVMs and Feature Selection
# Katie Abrahams, abrahake@pdx.edu
# 2/16/16

from sklearn import svm

from src.sklearn_svm.input import *

# As other classifiers, SVC, NuSVC and LinearSVC take as input two arrays:
# an array X of size [n_samples, n_features] holding the training samples,
# and an array y of class labels (strings or integers), size [n_samples]:
clf = svm.LinearSVC(C=.1)
#clf = svm.SVC(C=.1, kernel='linear')
clf.fit(X_scaled, X_col)
print clf

# Use TA's written tools
# main()
# features, labels = LoadSpamData()
# BalanceDataset(features, labels)
# # shuffle data
# # split into test and training
# # call for both training and test data
# features, labels = ConvertDataToArrays(features, labels)
# features = NormalizeFeatures(features)
# PrintDataToSvmLightFormat(features, labels)

# run SVMlight with
# HAL:src katieabrahams$ ./svm_learn spambase.data