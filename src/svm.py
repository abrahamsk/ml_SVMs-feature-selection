#!/usr/bin/env python
# -*- coding: utf-8 -*-
##  utf-8 for non-ASCII chars

# Machine Learning 445
# HW 3: SVMs and Feature Selection
# Katie Abrahams, abrahake@pdx.edu
# 2/16/16

from sklearn import svm
from hw3tools import *

# test
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
print clf

# Use TA's written tools
features, labels = LoadSpamData()
BalanceDataset(features, labels)
# shuffle data
# split into test and training
# call for both training and test data
features, labels = ConvertDataToArrays(features, labels)
features = NormalizeFeatures(features)
PrintDataToSvmLightFormat(features, labels)

# run SVMlight with
# HAL:src katieabrahams$ ./svm_learn spambase.data