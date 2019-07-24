"""Does cross validation using per class holdouts for a classifier."""

import common_python.constants as cn

import numpy as np
import pandas as pd


class CrossValidator(object):

  def __init__(self, classifier, df_X, ser_y):
    """
    :param Classifier classifier: fit, score methods
    :param pd.DataFrame df_X: columns of features, rows of instances
    :param pd.Series ser_y: state values
    Notes
      1. df_X, ser_y must have the same index
    """
    self.classifier = classifier
    self.df_X = df_X
    self.ser_y = ser_y

  def do(self, iterations=5, holdouts=1):
    """
    Does cross validation holding out 1 instance for each state.
    :param int interations: number of cross validations done
    :param int holdouts: number of instances per state in test data
    :return float, float: mean, std of accuracy for cross validations
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
    classes = self.ser_y.unique()
    indices = self.ser_y.index.tolist()
    for _ in range(iterations):
      # Construct test set
      indices = np.random.permutation(indices)
      df_X = sortIndex(self.df_X, indices)
      ser_y = sortIndex(self.ser_y, indices)
      test_indices = []
      for cls in classes:
        ser = ser_y[ser_y == cls]
        if len(ser) <= holdouts:
          raise ValueError("Class %s has fewer than %d holdouts" %
              (cls, holdouts))
        idx = ser.index[0:holdouts].tolist()
        test_indices.extend(idx)
      df_X_train, df_X_test = partitionData(df_X, indices, test_indices)
      ser_y_train, ser_y_test = partitionData(ser_y, indices, test_indices)
      self.classifier.fit(df_X_train, ser_y_train)
      score = self.classifier.score(df_X_test, ser_y_test)
      scores.append(score)
    return np.mean(scores), np.std(scores)
        
      
    
    
