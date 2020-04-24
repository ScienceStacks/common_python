"""
Selects features to use in a classifier.
The classifier may have multiple classes.

Technical notes:
1. cls is the name used for a class
"""

from common_python.classifier import util_classifier

import copy
import numpy as np
import pandas as pd

MAX_CORR = 0.5  # Maxium correlation with an existing feature


################### Base Class #################
class FeatureSelector(object):
  """
  Selects features for a class for a classifier
  by using features in order.
  FeatureSelector responsibilities
    1. feature_dct
       Container for features chosen for each class.
    2. add(cls)
       Updates feature_dct to include
       the next best feature for forward
       feature selection. This is done by eliminating
       from consideration features that are too highly
       correlated with features that are already selected.
    3. zeroValues(cls)
       Set to 0 the values of unselected features.
    4. remove(cls)
       Removes the last feature added
  """

  def __init__(self, df_X, ser_y):
    """
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: class
    :param float max_corr: maximum correlation
        between a new feature an an existing feature
    """
    # Private
    self._df_X = df_X
    self._ser_y = ser_y
    self._classes = list(self._ser_y.unique())
    # Public
    # Features selected for each state
    self.all_features = df_X.columns.tolist()
    self.feature_dct = {c: [] for c in self._classes}
    self.ordered_dct, self.fstat_dct = self.makeDct()
    self.remove_dct = {c: [] for c in self._classes}

  def orderFeatures(self, cls, df_fstat=None, ser_weight=None):
    """
    Constructs features ordered in descending priority
    and F-statistics for each class.
    :param object cls: class
    :param pd.DataFrame df_fstat: used, else computed
    :param pd.Series ser_weight: weights applied to instances
    :return dict, pd.DataFrame:
      dict
          key: class
          value: list of features by descending fstat
      pd.DataFrame
          index: feature
          column: class
          value: F-statistic
    """
    if df_fstat is None:
      df_fstat = util_classifier.makeFstatDF(
          self._df_X, self._ser_y, ser_weight=ser_weight)
    ser_fstat = df_fstat[cls]
    ser_fstat.sort_values()
    return df_fstat, ser_fstat.index.tolist()

  def makeDct(self, ser_weight=None):
    """
    Constructs features ordered in descending priority
    and F-statistics for each class.
    :return dict, dict:
        First:
          key: class
          value: list of features by descending fstat
        Second:
          key: class
          value: Series for fstat
    """
    ordered_dct = {}
    fstat_dct = {}
    df_fstat = None
    for cls in self._classes:
      df_fstat, ordered = self.orderFeatures(cls,
          df_fstat=df_fstat, ser_weight=ser_weight)
      ser_fstat = df_fstat[cls]
      ser_fstat.sort_values()
      fstat_dct[cls] = ser_fstat
      ordered_dct[cls] = ordered
    return ordered_dct, fstat_dct

  def zeroValues(self, cls):
    """
    Sets values of non-features to zero.
    :return pd.DataFrame: Non-feature columns are 0
    """
    df_X_sub = self._df_X.copy()
    non_features = list(set(self.all_features
        ).difference(self.feature_dct[cls]))
    df_X_sub[non_features] = 0
    return df_X_sub

  def add(self, cls, **kwargs):
    """
    Adds a feature for the class selecting
    the top feature not yet chosen.
    :param object cls:
    :return bool: True if a feature was added.
    """
    ordereds = [f for f in self.ordered_dct[cls]
        if (not f in self.feature_dct[cls]) and
        (not f in self.remove_dct[cls])]
    if len(ordereds) > 0:
      self.feature_dct[cls].append(ordereds[0])
      return True
    else:
      return False

  def remove(self, cls):
    """
    Removes the last feature added.
    :param object cls:
    """
    self.remove_dct[cls].append(
        self.feature_dct[cls][-1])
    self.feature_dct[cls] = self.feature_dct[cls][:-1]


########### Select based on correations #################
class FeatureSelectorCorr(FeatureSelector):
  """
  Selects features for a class using correlations.
  """

  def __init__(self, df_X, ser_y, max_corr=MAX_CORR):
    """
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: class
    :param float max_corr: maximum correlation
        between a new feature an an existing feature
    """
    super().__init__(df_X, ser_y)
    # Private
    self._max_corr = max_corr
    self._df_corr = self._df_X.corr()
    # Public
    # Features in descending order
    # f-statistics for features by class
    self.ordered_dct, self.fstat_dct = self.makeDct()

  def add(self, cls):
    """
    Adds a feature for the class.
    :param object cls:
    :return bool: True if a feature was added.
    """
    if len(self.feature_dct[cls]) > 0:
      # Not the first feature
      df_corr = copy.deepcopy(self._df_corr)
      df_corr = df_corr[self.feature_dct[cls]]
      ser_max = df_corr.max(axis=1)
      ser_max = ser_max.apply(lambda v: np.abs(v))
      # Choose the highest priority feature that is
      # not highly correlated with the existing features.
      indices = ser_max.index[ser_max < self._max_corr]
      ordered = list(set(
          self.ordered_dct[cls]).difference(
          self.remove_dct[cls]))
      feature_subset = [f for f in ordered
          if f in ser_max[indices]]
    else:
      # Handle first feature
      feature_subset = self.ordered_dct[cls]
    if len(feature_subset) > 0:
      self.feature_dct[cls].append(feature_subset[0])
      return True
    else:
      return False


####### Select based on classification residuals #######
class FeatureSelectorResidual(FeatureSelector):
  """
  Selects features for a class using differences in classification.
  """

  def __init__(self, df_X, ser_y, weight=None):
    """
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: class
    :param float max_corr: maximum correlation
        between a new feature an an existing feature
    :param float weight: weight given to misclassifications
        default is 1 + ratio of non-misses to misses
    """
    super().__init__(df_X, ser_y)
    self._weight = weight

  def add(self, cls, ser_pred=None):
    """
    Adds a feature for the class.
    :param object cls:
    :param pd.Series ser_pred: predicted classes
    :return bool: True if a feature was added.
    """
    indices_miss = self._ser_y.index[self._ser_y != ser_pred]
    ser_weight = self._ser_y.copy()
    ser_weight[:] = 1
    if self._weight is None:
      weight = 1 + (len(self._ser_y)  \
      - len(indices_miss)) / len(indices_miss)
    else:
      weight - self._weight
    ser_weight.loc[indices_miss] = weight
    _, ordereds = self.orderFeatures(cls,
        ser_weight=ser_weight)
    ordereds = [f for f in ordereds
        if (not f in self.feature_dct[cls]) and
        (not f in self.remove_dct[cls])]
    if len(ordereds) == 0:
      return False
    else:
      self.feature_dct[cls].append(ordereds[0])
      return True
