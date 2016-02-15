#!/usr/bin/env python
# -*- coding: utf-8 -*-
##  utf-8 for non-ASCII chars

# Machine Learning 445
# HW 3: SVMs and Feature Selection
# Katie Abrahams, abrahake@pdx.edu
# 2/16/16

import pandas as pd
from sklearn import preprocessing
import numpy as np

df = pd.read_csv('/Users/katieabrahams/PycharmProjects/machinelearningHW3/src/spambase/spambase.data', header=None)
# print df  #[4601 rows x 58 columns]
# df.to_csv("df.csv")

# Create a subset of the data that has equal numbers of positive and negative examples
# print df[57]
# count of pos/neg
# print df[57].value_counts()
# 0: 2788
# 1: 1813

# Create a subset of the data that has equal numbers of positive and negative examples (1813 each of pos/neg) #
df_pos = (df.loc[df[57] == 1])
# use size 1812 for division with no remainder
# when dividing into test and training data
df_pos = df_pos[0:1812].reset_index(drop=True)
# print df_pos  # [1812 rows x 58 columns]
df_neg = (df.loc[df[57] == 0])
# create a dataframe for negative instances equal in size to positive dataframe
df_neg = df_neg[0:1812].reset_index(drop=True)
# print df_neg  #[1812 rows x 58 columns]

###########################################################################

# concat pos + neg dataframe subsets for training and test data, 906 rows of each

# training data
frames_training = [df_pos[0:905], df_neg[0:905]]
df_training = pd.concat(frames_training)
df_training = df_training.reset_index(drop=True)
# print df_training  # [1810 rows x 58 columns]

# test data
frames_test = [df_pos[906:1811], df_neg[906:1811]]
df_test = pd.concat(frames_test)
df_test = df_test.reset_index(drop=True)
# print df_test  # [1810 rows x 58 columns]

###########################################################################

# shuffle training data #
# frac=1 means return all rows in random order
df_training = df_training.sample(frac=1).reset_index(drop=True)
# print df_training

###########################################################################

# scale data #
# X_scaled = preprocessing.scale(df_training[0:57])
# frames = [X_scaled, df_training[58]]
# X_scaled = pd.concat(frames)

# convert dataframe into a numpy matrix
X = df_training.as_matrix().astype(np.float)
# scale data #
# preprocess everything in training data matrix except the last column (1 or 0 to identify spam or not)
# then concat the identifying column to the preprocessed training data
# print X.shape  # (1810, 58)
scaler = preprocessing.StandardScaler()
X_to_scale = X[:,:57].copy()
# print X_to_scale.shape  # (1810, 57)
X_scaled = scaler.fit_transform(X_to_scale)
# print X_scaled.mean(axis=0)
# print X_scaled
# print X_scaled.shape  # (1810, 57)
X_col = X[:,57]
# print X_col[None].T.shape  # (1810, 1)
X_concat = np.concatenate((X_scaled, X_col[None].T), axis=1)
# print X_concat.shape  # (1810, 58)
np.savetxt("/Users/katieabrahams/PycharmProjects/machinelearningHW3/src/sklearn_svm/numpy_train.csv", X_concat, delimiter=",")

# Scale test data using standardization parameters from training data
X_test = df_test.as_matrix().astype(np.float)
# print X_test
X_test_to_scale = X_test[:,:57].copy()
X_test_scaled = scaler.fit_transform(X_test_to_scale)
X_test_col = X_test[:,57]
X_test_concat = np.concatenate((X_test_scaled, X_test_col[None].T), axis=1)
np.savetxt("/Users/katieabrahams/PycharmProjects/machinelearningHW3/src/sklearn_svm/numpy_test.csv", X_test_concat, delimiter=",")



