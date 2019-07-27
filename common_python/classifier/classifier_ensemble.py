"""Maniuplations of an ensemble of classifiers for same data."""

import common_python.constants as cn

import collections
import copy
import numpy as np
import pandas as pd


CrossValidationResult = collections.namedtuple(
    "CrossValidationResult", "mean std ensemble")


class ClassifierEnsemble(object):

  def __init__(self, classifiers):
    """
    :param list-Classifier classifiers: classifiers
    """
    self.classifiers = classifiers

  @classmethod
  def crossVerify(cls, classifier, df_X, ser_y,
      iterations=5, holdouts=1):
    """
    Does cross validation wth holdouts for each state.
    :param Classifier classifier: untrained classifier with fit, score methods
    :param pd.DataFrame df_X: columns of features, rows of instances
    :param pd.Series ser_y: state values
    :param int interations: number of cross validations done
    :param int holdouts: number of instances per state in test data
    :return CrossValidationResult:
    Notes
      1. df_X, ser_y must have the same index
    """
    def sortIndex(container, indices):
      container = container.copy()
      container.index = indices
      return container.sort_index()
    def partitionData(container, all_indices, test_indices):
      train_indices = list(set(all_indices).difference(test_indices))
      if isinstance(container, pd.DataFrame):
        container_test = container.loc[test_indices, :]
        container_train = container.loc[train_indices, :]
      else:
        container_test = container.loc[test_indices]
        container_train = container.loc[train_indices]
      return container_train, container_test
    #
    scores = []
    classifiers = []
    classes = ser_y.unique()
    indices = ser_y.index.tolist()
    for _ in range(iterations):
      # Construct test set
      new_classifier = copy.deepcopy(classifier)
      classifiers.append(new_classifier)
      indices = np.random.permutation(indices)
      df_X = sortIndex(df_X, indices)
      ser_y = sortIndex(ser_y, indices)
      test_indices = []
      for cl in classes:
        ser = ser_y[ser_y == cl]
        if len(ser) <= holdouts:
          raise ValueError("Class %s has fewer than %d holdouts" %
              (cl, holdouts))
        idx = ser.index[0:holdouts].tolist()
        test_indices.extend(idx)
      df_X_train, df_X_test = partitionData(df_X, indices, test_indices)
      ser_y_train, ser_y_test = partitionData(ser_y, indices, test_indices)
      new_classifier.fit(df_X_train, ser_y_train)
      score = new_classifier.score(df_X_test, ser_y_test)
      scores.append(score)
    return CrossValidationResult(
        mean=np.mean(scores), 
        std=np.std(scores), 
        ensemble=cls(classifiers)
        )
        
   
class LinearSVMEnsemble(ClassifierEnsemble):

  pass   
    
    
