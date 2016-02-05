#!/usr/bin/env python
# -*- coding: utf-8 -*-
##  utf-8 for non-ASCII chars

# Machine Learning 445
# HW 3: SVMs and Feature Selection
# Katie Abrahams, abrahake@pdx.edu
# 2/16/16

from sklearn import svm

# test
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
print clf