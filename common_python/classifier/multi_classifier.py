"""
Classifier for data with multiple classes. The
classifier is constructed from a binary classier,
which is referred to as the base classifier.
Classifier features are selected using a
FeatureHandler instance.

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
class FeatureHandler(object):
  """
  Default handler for selecting features for a class.
  FeatureHandler responsibilities
    1. Create descending order of features for each class
    2. Select the next best feature for forward
       feature selection. This is done by eliminating
       from consideration features that are too highly
       correlated with features that are already selected.
  """

  def __init__(self, df_X, ser_y, max_corr=0.5):
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
    # Public
    self.df_X = df_X
    self.ser_y = ser_y
    self._classes = list(self.ser_y.uniquie())
    self.all_features = df_X.columns.tolist()
    # Features selected for each state
    self.feature_dct = {c: [] for c in self._classes}
    # Private
    # Features in descending priority order by class
    self._max_corr = max_corr
    self._ordered_dct = {}
    self._df_corr = np.corrcoef(self.df_X)

  @property
  def ordered_dct(self):
    """
    Features ordered in descending priority for each class.
    :return dict:
        key: class
        value: list of features in descending order
    """
    if len(self._ordered_dct) == 0:
      df_fstat = util_classifier.makeFstatDF(
          self.df_X, self.ser_y)
      for cls in self._classes:
        ser_fstat = df_fstat[cls]
        ser_fstat.sort_values()
        self._ordered_dct[cls] = ser_fstat.index.tolist()
    return self._ordered_dct

  def getNonFeatures(self, cls):
    return list(set(self.feature_dct[cls]).difference(
        features))

  def setNonFeatures(self, cls):
    """
    Sets values of non-features to zero.
    :return pd.DataFrame: Non-feature columns are 0
    """
    df_X_sub = self.df_X.copy()
    non_features = list(set(self.feature_dct[cls]
        ).difference(self.feature_dct[cls]))
    df_X_sub[non_features] = 0
    return df_X_sub

  def nextFeatures(self, cls):
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
    feature_subset = [f for f in self._ordered_dct[cls]
        if f in ser_max[sel]]
    if len(feature_subset) > 0:
      self.feature_dct[cls].append(feature_subset[0])
      return self.feature_dct[cls]
    else:
      return None


class MultiClassifier(object):
 
  def __init__(self, base_clf=svm.LinearSVC(),
        feature_handler_cls=FeatureHandler,
        desired_accuracy=0.9,
        **kwargs):
    """
    :param Classifier base_clf: 
    :param FeatureHandler feature_handler:
    :param float desired_accuracy: accuracy for clf
    :param dict kwargs: arguments passed to FeatureHandler
    """
    # Public
    self.feature_dct = {}  # Features used for classifier
    self.clf_dct = {}  # key is cls; value is clf
    self.score_dct = {}  # key is cls; value is score
    self.feature_handler = None
    # Private
    self._kwargs = kwargs
    self._desired_accuracy = desired_accuracy
    self.feature_handler_cls = feature_handler_cls
    self._base_clf = base_clf
    self._classes = []

  def fit(self, df_X, ser_y):
    """
    Selects the top features for each class and fits
    a classifier with the desired accuracy.
    Classifiers are in self.clfs.
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: class
    :globals:
        assigned: all_features, classes,
                  clf_dct, score_dct, feature_dct
    """
    self.feature_handler = self.feature_handler_cls(
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
        if self.feature_handler.nextFeatures(cls) is None:
          break
        df_X_rank = self.feature_handler.setNonFeatures(cls)
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
    :globals:
        read: clf_dct
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
