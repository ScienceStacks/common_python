'''Selects features based on classifier results.'''

from common_python.classifier import util_classifier

import copy
import numpy as np
import pandas as pd

CLASSES = [0, 1]
MAX_CORR = 0.5  # Maxium correlation with an existing feature


class BinaryFeatureSelector(object):
  """
  Does feature selection for binary classes.
  by using features in order.
  FeatureSelector responsibilities
    1. feature_manager - object that contains features
    2. score - score achieved
    3  best_score
    4. identifier - identifies this instance
    5. is_done - indicates have completed.
  This is a computationally intensive activity and so
  the implementation allows for restarts.
  """

  def __init__(self, df_X, ser_y,
      checkpoint_function,
      feature_manager_class,
      classifier,
      min_incr_score=MIN_INCR_SCORE,
      max_iter=MAX_ITER, 
      max_degrade=MAX_DEGRADE,
      ):
    """
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: binary class values (0, 1)
    :param type-FeatureManager feature_manager_class:
    :param Classifier classifier:
    :param float min_incr_score: min amount by which
        a feature must increase the score to be included
    :param int max_iter: maximum number of iterations
    :param float max_degrade: maximum difference between
        best score and actual
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

  def add(self, cls, feature=None, **kwargs):
    """
    Adds a feature for the class selecting
    the top feature not yet chosen.
    :param object cls:
    :param object feature: specific feature to add
    :return bool: True if a feature was added.
    """
    if feature is None:
      feature = self.findFeature(cls, **kwargs)
    if feature is not None:
      self.feature_dct[cls].append(feature)
      return True
    else:
      return False

  def findFeature(self, cls, **kwargs):
    """
    Adds a feature for the class selecting
    the top feature not yet chosen.
    :param object cls:
    :return object: feature to add
    Should be overridden is in subclasses
    """
    ordereds = [f for f in self.ordered_dct[cls]
        if (not f in self.feature_dct[cls]) and
        (not f in self.remove_dct[cls])]
    if len(ordereds) > 0:
      feature = ordereds[0]
    else:
      feature = None
    return feature

  def remove(self, cls, feature=None):
    """
    Removes a specified feature, or the
    last one added if none is specified.
    :param object cls:
    :param object feature:
    """
    if feature is None:
      feature = self.feature_dct[cls][-1]
    self.remove_dct[cls].append(feature)
    self.feature_dct[cls].remove(feature)


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

  def findFeature(self, cls, **kwargs):
    """
    Finds feature to add for class.
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
      feature = feature_subset[0]
    else:
      feature = None
    return feature


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

  def findFeature(self, cls, ser_pred=None):
    """
    Finds feature to add.
    :param object cls:
    :param pd.Series ser_pred: predicted classes
    :return object: feature
    """
    indices_miss = self._ser_y.index[
        self._ser_y != ser_pred]
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
      feature = None
    else:
      feature = ordereds[0]
    return feature
