"""Classification done by an ensemble of SVM classifiers."""

import common_python.constants as cn
from common_python.util import util
from common_python.classifier.classifier_ensemble ClassifierEnsemble
from common_python.classifier.classifier_collection ClassifierCollection

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm


class SVMEnsemble(ClassifierEnsemble):
  """
  A classifier formed by an ensemble of SVM classifiers. Does
  feature selection based on the rank of the full set of features.
  """
 
  def __init__(self, clf, filter_high_rank=None, holdouts=1, size=1):
    """
    :param int/None filter_high_rank: maxium rank considered
    :param int holdouts: number of holdouts when fitting
    :param int size: size of the ensemble
    """
    self.base_clf = clf
    self.filter_high_rank = filter_high_rank
    self.holdouts = holdouts
    self.size = size
    super().__init__(None, None, None, None)

  def fit(self, df_X, ser_y):
    """
    Fits the number of classifiers desired to the data with
    holdouts. Current selects holdouts independent of class.
    :param pd.DataFrame df_X: feature vectors; indexed by instance
    :param pd.Series ser_y: classes; indexed by instance
    """
    collection = ClassifierCollection.makeByRandomHoldout(
        self.base_clf, df_X, ser_y, self.size, 
        holdouts=self.holdouts)
    self.update(collection)
    if self.filter_high_rank is None:
      return result
    # Select the features
    df_rank = self.makeRankDF()
    df_rank_sub = df_rank.loc[df_rank.index[0:filter_high_rank], :]
    columns = df_rank_sub.index.tolist()
    df_X_sub = df_X[columns]
    collection = ClassifierCollection.makeByRandomHoldout(
        self.base_clf, df_X, ser_y, self.size, 
        holdouts=self.holdouts)
    self.update(collection)

  def _orderFeatures(self, clf, class_selection):
    """
    Orders features by descending value of importance.
    :param int class_selection: restrict analysis to a single class
    :return list-int:
    """
    if class_selection is None:
      coefs = [max([np.abs(x) for x in xv]) for xv in zip(*clf.coef_)]
    else:
      coefs = [np.abs(x) for x in clf.coef_[class_selection]]
    length = len(coefs)
    sorted_tuples = np.argsort(coefs).tolist()
    # Calculate rank in descending order
    result = [length - sorted_tuples.index(v) for v in range(length)]
    return result

  def makeRankDF(self, class_selection=None):
    """
    Constructs a dataframe of feature ranks for importance.
    A more important feature has a lower rank (closer to 0)
    :param int class_selection: restrict analysis to a single class
    :return pd.DataFrame: columns are cn.MEAN, cn.STD, cn.STERR
    """
    df_values = pd.DataFrame()
    for idx, clf in enumerate(self.classifiers):
      df_values[idx] = pd.Series(self._orderFeatures(clf, class_selection),
          index=self.features)
    df_result = self._makeFeatureDF(df_values)
    return df_result.sort_values(cn.MEAN)

  def makeImportanceDF(self, class_selection=None):
    """
    Constructs a dataframe of feature importances.
    :param int class_selection: restrict analysis to a single class
    :return pd.DataFrame: columns are cn.MEAN, cn.STD, cn.STERR
    The importance of a feature is the maximum absolute value of its coefficient
    in the collection of classifiers for the multi-class vector (one vs. rest, ovr).
    """
    ABS_MEAN = "abs_mean"
    df_values = pd.DataFrame()
    for idx, clf in enumerate(self.classifiers):
      if class_selection is None:
        values = [max(np.abs(x) for x in xv) for xv in  zip(*clf.coef_)]
      else:
        values = [x for x in clf.coef_[class_selection]]
      df_values[idx] = pd.Series(values, index=self.features)
    df_result = self._makeFeatureDF(df_values)
    df_result[ABS_MEAN] = [np.abs(x) for x in df_result[cn.MEAN]]
    df_result = df_result.sort_values(ABS_MEAN, ascending=False)
    del df_result[ABS_MEAN]
    return df_result
