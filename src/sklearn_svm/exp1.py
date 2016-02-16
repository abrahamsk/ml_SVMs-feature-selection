#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Machine Learning 445
# HW 3: SVMs and Feature Selection
# Katie Abrahams, abrahake@pdx.edu
# 2/16/16
from __future__ import division  # float division
from input import *
import sys
from sklearn import svm, cross_validation, metrics
from sklearn.metrics import recall_score, accuracy_score, roc_curve, auc
# ROC curve:
import matplotlib.pyplot as plot
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

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
zero = (.1 * 10**-100)

c_params = [zero, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
# 1) Split training data
validation_set = []
validation_set_classification = []
training_set = []
training_set_classifications = []

# print X_col.shape # 1810

# For each distinct subset, take a different 10% portion of the training data
# to set aside for validation.  The remaining 90% will be used to train the SVM
# Set 0
validation_set.append(X_scaled[0:181])
validation_set_classification.append(X_col[0:181])
training_set.append(X_scaled[181:])
training_set_classifications.append(X_col[181:])
# 1
validation_set.append(X_scaled[181:362])
validation_set_classification.append(X_col[181:362])
training_set.append(np.concatenate((X_scaled[0:181], X_scaled[362:1810]), axis=0))
training_set_classifications.append(np.concatenate((X_col[0:181, None], X_col[362:1810,None]), axis=0))
# 2
validation_set.append(X_scaled[362:543])
validation_set_classification.append(X_col[362:543])
training_set.append(np.concatenate((X_scaled[0:362], X_scaled[543:1810]), axis=0))
training_set_classifications.append(np.concatenate((X_col[0:362, None], X_col[543:1810,None]), axis=0))
# 3
validation_set.append(X_scaled[543:724])
validation_set_classification.append(X_col[543:724])
training_set.append(np.concatenate((X_scaled[0:543], X_scaled[724:1810]), axis=0))
training_set_classifications.append(np.concatenate((X_col[0:543, None], X_col[724:1810,None]), axis=0))
# 4
validation_set.append(X_scaled[724:905])
validation_set_classification.append(X_col[724:905])
training_set.append(np.concatenate((X_scaled[0:724], X_scaled[905:1810]), axis=0))
training_set_classifications.append(np.concatenate((X_col[0:724, None], X_col[905:1810,None]), axis=0))
# 5
validation_set.append(X_scaled[905:1086])
validation_set_classification.append(X_col[905:1086])
training_set.append(np.concatenate((X_scaled[0:905], X_scaled[1086:1810]), axis=0))
training_set_classifications.append(np.concatenate((X_col[0:905, None], X_col[1086:1810,None]), axis=0))
# 6
validation_set.append(X_scaled[1086:1267])
validation_set_classification.append(X_col[1086:1267])
training_set.append(np.concatenate((X_scaled[0:1086], X_scaled[1267:1810]), axis=0))
training_set_classifications.append(np.concatenate((X_col[0:1086, None], X_col[1267:1810,None]), axis=0))
# 7
validation_set.append(X_scaled[1267:1448])
validation_set_classification.append(X_col[1267:1448])
training_set.append(np.concatenate((X_scaled[0:1267], X_scaled[1448:1810]), axis=0))
training_set_classifications.append(np.concatenate((X_col[0:1267, None], X_col[1448:1810,None]), axis=0))
# 8
validation_set.append(X_scaled[1448:1629])
validation_set_classification.append(X_col[1448:1629])
training_set.append(np.concatenate((X_scaled[0:1448], X_scaled[1629:1810]), axis=0))
training_set_classifications.append(np.concatenate((X_col[0:1448, None], X_col[1629:1810,None]), axis=0))
# 9
validation_set.append(X_scaled[1629:1810])
validation_set_classification.append(X_col[1629:1810])
training_set.append(X_scaled[0:1629])
training_set_classifications.append(X_col[0:1629])

##########################################################################################################

# 2)  For each value of the C parameter j:
# store avg accuracies to compute the best
avg_accuracy = []
for j in c_params:
    accuracy = []
    for i in xrange(10):
        # Select Si to be the validation set: validation_set[i]
        # Train linear SVM using C=j and all training data except Si.
        clf = svm.SVC(C=j, kernel='linear')
        # fit the SVM model according to the given training data.  fit() returns self.
        clf.fit(training_set[i], training_set_classifications[i].ravel())
        accuracy.append(clf.score(validation_set[i], validation_set_classification[i]))
        text = "\rC parameter: "+str((j))#+"\r xrange "+str((i)+1)
        sys.stdout.write(text)
    # 3) Compute average accuracy for C=j
    avg_accuracy.append(sum(accuracy) / len(accuracy))
# 4) Choose value C=C* that results in highest Aj
winner = max(avg_accuracy)
winner_index = avg_accuracy.index(winner)

# 5) Train new linear SVM using all training data with C=C*
print "\nTraining new linear SVM on all training data using C*..."
model = svm.SVC(C=winner, kernel='linear')
# Fit the SVM model according to the given training data.
model.fit(X_scaled, X_col)
print "done!"

# 6) Test learned SVM model on test data. Report accuracy, precision, and recall
# (using threshold 0 to determine positive and negative classifications)
print "Testing learned SVM... "
# score(X, y): X Test samples, y True labels for X
# Returns: Mean accuracy of self.predict(X) wrt. y
# return the mean accuracy on the given test data and labels
acc = model.score(X_test_scaled, X_test_col)
# Predict class labels for samples in X_test_scaled.
predicted = model.predict(X_test_scaled)
print "done!"

def print_stats():
    """
    Print stats about experiment 1 data
    :return nothing, but print stats:
    """
    print "----- Exp 1 Accuracy -----\n",acc,"\n--------------------------"

    # Performance metrics
    # Returns text summary of the precision, recall, F1 score for each class
    target_names = ['Not spam', 'Spam']
    print "Classification report for %s" % model
    print metrics.classification_report(X_test_col, predicted, target_names=target_names)

    print "Confusion matrix:"
    print metrics.confusion_matrix(X_test_col, predicted)

##########################################################################################################

# 7) ROC Curve
# sourced from:
# scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#example-model-selection-plot-roc-py
def plot_roc():
    """
    Plot ROC curve for experiment 1
    :return nothing, but Python displays a plotted curve:
    """
    data = np.concatenate((X_scaled, X_test_scaled), axis=0)
    classifications = np.concatenate((X_col[:, None], X_test_col[:, None]), axis=0)

    # Binarize the output
    classifications = label_binarize(classifications, classes=[0, 1])
    n_classes = classifications.shape[1]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, classifications, test_size=.5, random_state=0)
    # Learn to predict each class against the other
    # Fit one classifier per class. For each classifier, the class is fitted against all the other classes.
    classifier = OneVsRestClassifier(svm.SVC(C=winner, kernel='linear', probability=True))
    # Fit the model according to the given training data
    # Predict confidence scores for samples:
    # The confidence score for a sample is the signed distance of that sample to the hyperplane
    # Returns:
    # Confidence scores per (sample, class) combination. In the binary case, confidence score for
    # self.classes_[1] where >0 means this class would be predicted
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    false_pos_rate = dict()
    true_pos_rate = dict()
    roc_auc = dict()
    for i in range(n_classes):
        false_pos_rate[i], true_pos_rate[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(false_pos_rate[i], true_pos_rate[i])

    # Compute micro-average ROC curve and ROC area
    false_pos_rate["micro"], true_pos_rate["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(false_pos_rate["micro"], true_pos_rate["micro"])


    ################################################
    # Plot ROC curves for multiclass
    # Compute macro-average ROC curve and ROC area
    # macro-averaging gives equal weight to the classification of each label

    # Aggregate all false positive rates
    all_false_pos_rate = np.unique(np.concatenate([false_pos_rate[i] for i in range(n_classes)]))

    # Interpolate all ROC curves at points
    mean_true_pos_rate = np.zeros_like(all_false_pos_rate)
    for i in range(n_classes):
        mean_true_pos_rate += interp(all_false_pos_rate, false_pos_rate[i], true_pos_rate[i])

    # Average and compute AUC
    mean_true_pos_rate /= n_classes
    false_pos_rate["macro"] = all_false_pos_rate
    true_pos_rate["macro"] = mean_true_pos_rate
    roc_auc["macro"] = auc(false_pos_rate["macro"], true_pos_rate["macro"])

    # Plot all ROC curves
    plot.figure()
    plot.plot(false_pos_rate["micro"], true_pos_rate["micro"],
              label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]), linewidth=2)

    plot.plot(false_pos_rate["macro"], true_pos_rate["macro"],
              label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]), linewidth=2)

    for i in range(n_classes):
        plot.plot(false_pos_rate[i], true_pos_rate[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    # Display plot
    plot.plot([0, 1], [0, 1], 'k--')
    plot.xlim([0.0, 1.0])
    plot.ylim([0.0, 1.05])
    # label x and y axes, generate title
    plot.xlabel('False Positive Rate')
    plot.ylabel('True Positive Rate')
    plot.title('ROC (Receiver operating characteristic) curve, multi-class')
    plot.legend(loc="lower right")
    # show plotted ROC curve
    plot.show()

##########################################################################################################

# Only print exp1 stats and plot the ROC curve if exp1 is the main program
def main():
    print_stats()
    plot_roc()

if __name__ == "__main__":
  main()