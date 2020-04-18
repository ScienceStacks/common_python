"""
Classifier for data with multiple classes. The
classifier is constructed from a binary classier,
which is referred to as the base classifier.
Classifier features are selected using a
FeatureSelector instance.

The base classifier must expose methods for fit,
predict, score. Fitting is done by forward selection
of features to achieve a desired accuracy.
This is a "soft" constraint, and it is checked
without the use of cross validation (for performance
reasons).

Technical notes:
1. cls is the name used for a class
"""

from common_python.classifier import util_classifier

import copy
import numpy as np
import pandas as pd
from sklearn import svm

MAX_CORR = 0.5  # Maxium correlation with an existing feature


###################################################
class FeatureSelector(object):
  """
  Selects features for a class for a classifier.
  FeatureSelector responsibilities
    1. feature_dct
       Container for features chosen for each class.
    2. getNextFeatures()
       Updates feature_dct to include
       the next best feature for forward
       feature selection. This is done by eliminating
       from consideration features that are too highly
       correlated with features that are already selected.
    3. getNonFeatures()
       Compute the complement of the current features
       and the full set of features.
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
    # Private
    self._df_X = df_X
    self._ser_y = ser_y
    self._classes = list(self._ser_y.uniquie())
    self._max_corr = max_corr
    self._df_corr = np.corrcoef(self._df_X)
    # Public
    # Features in descending order
    # f-statistics for features by class
    self.ordered_dct, self.fstat_dct = self._makeOrderedDct()
    # Features selected for each state
    self.feature_dct = {c: [] for c in self._classes}

  def _makeDct(self):
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
    df_fstat = util_classifier.makeFstatDF(
        self._df_X, self._ser_y)
    for cls in self._classes:
      ser_fstat = df_fstat[cls]
      ser_fstat.sort_values()
      fstat_dct[cls] = ser_fstat
      ordered_dct[cls] = ser_fstat.index.tolist()
    return ordered_dct, fstat_dct

  def getNonFeatures(self, cls):
    """
    Sets values of non-features to zero.
    :return pd.DataFrame: Non-feature columns are 0
    """
    df_X_sub = self._df_X.copy()
    non_features = list(set(self.feature_dct[cls]
        ).difference(self.feature_dct[cls]))
    df_X_sub[non_features] = 0
    return df_X_sub

  def getNextFeatures(self, cls):
    """
    Selects the next feature to add for this class.
    :param object cls:
    :return str: list-feature
    """
    df_corr = copy.deepcopy(self.df_corr)
    df_corr = df_corr[self.feature_dct[cls]]
    ser_max = df_corr.max(axis=1)
    ser_max = ser_max.apply(lambda v: np.abs(v))
    # Choose the highest priority feature that is
    # not highly correlated with the existing features.
    indices = ser_max.index[ser_max < self.max_corr]
    feature_subset = [f for f in self.ordered_dct[cls]
        if f in ser_max[sel]]
    if len(feature_subset) > 0:
      self.feature_dct[cls].append(feature_subset[0])
      return self.feature_dct[cls]
    else:
      return None


class MultiClassifier(object):
 
  def __init__(self, base_clf=svm.LinearSVC(),
        feature_selector_cls=FeatureSelector,
        desired_accuracy=0.9,
        **kwargs):
    """
    :param Classifier base_clf: 
    :param FeatureSelector feature_selector:
    :param float desired_accuracy: accuracy for clf
    :param dict kwargs: arguments passed to FeatureSelector
    """
    # Public
    self.clf_dct = {}  # key is cls; value is clf
    self.score_dct = {}  # key is cls; value is score
    self.feature_selector = None  # Constructed during fit
    # Private
    self._kwargs = kwargs
    self._desired_accuracy = desired_accuracy
    self.feature_selector_cls = feature_selector_cls
    self._base_clf = base_clf
    self._classes = []

  def fit(self, df_X, ser_y):
    """
    Selects the top features for each class and fits
    a classifier with the desired accuracy by including
    more features as selected by FeatureSelector.
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: class
    ASSIGNED INSTANCE VARIABLES
        _classes, clf_dct, score_dct
    """
    self.feature_selector = self.feature_selector_cls(
        df_X, ser_y, **self._kwargs)
    self._classes = ser_y.unique()
    for cls in self._classes:
      # Initialize for this class
      self.clf_dct[cls] = copy.deepcopy(self._base_clf)
      ser_y_cls = util_classifier.makeOneStateSer(ser_y,
          cls)
      last_score = 0
      # Use enough features to obtain the desired accuracy
      # This may not be possible
      for rank in range(len(features_cls)):
        if self.feature_selector.getNextFeatures(cls) is None:
          break
        df_X_rank = self.feature_selector.getNonFeatures(cls)
        self.clf_dct[cls].fit(df_X_rank, ser_y_cls)
        self.score_dct[cls] = self.clf_dct[cls].score(
            df_X, ser_y_cls)
        if self.score_dct[cls] >= self._desired_accuracy:
          break
      self.clf_dct[cls] = clf

  def predict(self, df_X):
    """
    Predict using the results of the classifier for each
    class. Prediction probability is the fraction of 
    classes that predict a 1 for the instance.
    :param pd.DataFrame: features, indexed by instance.
    :return pd.DataFrame: fraction prediction by class;
        column: class
        index: instance
        values: fraction
    """
    df_pred = pd.DataFrame()
    for cls in self._classes:
      df_pred[cls] = self.clf_dct[cls].predict(df_X)
    ser_tot = df_pred.sum(axis=1)
    df_pred = df_pred/ser_tot
    return df_pred

  def score(self, df_X, ser_y):
    """
    Evaluates the accuracy of the ensemble classifier for
    the instances pvodied.
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: class
    """
    score = 0
    total = 0
    df_predict = self.predict(df_X)
    for cls in self._classes:
      sel = ser_y == cls
      df_predict_sub = df_predict.loc[sel, :]
      score += df_predict_sub[cls].sum()
      total += len(df_pred_sub)
    if total != len(df_X):
      raise RuntimeError("Should process each row once.")
    return score/len(df_X)
