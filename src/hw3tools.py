"""
A few python functions for HW3: SVM Experiments  
CS 445 : Machine Learning
Written by Jordan Witte

2/8/16 

"""

import numpy as np

def LoadSpamData(filename = "spambase.data"):
  """
  Each line in the datafile is a csv with features values, followed by a single label (0 or 1), per sample; one sample per line
  """
  
  "The file function reads the filename from the current directory, unless you provide an absolute path e.g. /path/to/file/file.py or C:\\path\\to\\file.py"
  
  unprocessed_data_file = file(filename,'r')

  "Obtain all lines in the file as a list of strings."

  unprocessed_data = unprocessed_data_file.readlines()

  labels = []
  features = []

  for line in unprocessed_data:
    feature_vector = []

    "Convert the String into a list of strings, being the elements of the string separated by commas"
    split_line = line.split(',')

    "Iterate across elements in the split_line except for the final element "
    for element in split_line[:-1]:
      feature_vector.append(float(element))

    "Add the new vector of feature values for the sample to the features list"
    features.append(feature_vector)
    
    "Obtain the label for the sample and add it to the labels list"
    labels.append(int(split_line[-1]))

  "Return List of features with its list of corresponding labels"
  return features, labels


def BalanceDataset(features, labels):
  """
  Assumes the lists of features and labels are ordered such that all like-labelled samples are together (all the zeros come before all the ones, or vice versa)
  """
  
  count_0 = labels.count(0)
  count_1 = labels.count(1)
  balanced_count = min(count_0, count_1)
  

  #Indexing with a negative value tracks from the end of the list
  return features[:balanced_count] + features[-balanced_count:], labels[:balanced_count] + labels[-balanced_count:]
  


def ConvertDataToArrays(features, labels):
  """
  conversion to a numpy array is easy if you're starting with a List of lists.
  The returned array has dimensions (M,N), where M is the number of lists and N is the number of 
  
  """

  return np.asarray(features), np.asarray(labels)
  

def NormalizeFeatures(features):
  """
  I'm providing this mostly as a way to demonstrate array operations using Numpy.  Incidentally it also solves a small step in the homework.
  """
  
  "selecting axis=0 causes the mean to be computed across each feature, for all the samples"
  means = np.mean(features, axis = 0)
  
  std_devs = np.std(features, axis = 0)

  "Operations in numpy performed on a 2D array and a 1D matrix will automatically broadcast correctly, if the leading dimensions match."
  features = features - means
  # features -= means

  features /= std_devs

  return features


def PrintDataToSvmLightFormat(features, labels, filename = "svm_features.data"):
  """  
  Readable format for SVM Light should be, with 
  lable 0:feature0, 1:feature1, 2:feature2, etc...
  where label is -1 or 1.
  """

  if len(features) != len(labels):
    raise Exception("Number of samples and labels must match")
  dat_file = file(filename,'w')
  for s in range(len(features)):

    if labels[s]==0:
      line="-1 "
    else:
      line="1 "

    for f in range(len(features[s])):
      line +="%i:%f " % (f+1 , features[s][f])
    line += "\n"
    dat_file.write(line)
  dat_file.close()


def FeatureSubset(features_array, indices):
  """
  Takes the original set of features and returns a small array containing only the features with the given indices.

  features_array is a numpy 2D array of dimension (M,N), where M is the number of samples and N is the number of features in the feature vector.

  indices are those of the features to be used, as a list of integers  
  """
  return features_array[:,indices]

def main():

  features, labels = LoadSpamData()
  features, labels = BalanceDataset(features, labels)
  features, labels = ConvertDataToArrays(features, labels)
  features = NormalizeFeatures(features)
  indices = [0,1,2]
  features = FeatureSubset(features, indices)

  PrintDataToSvmLightFormat(features, labels)

if __name__ == "__main__":
  main()


