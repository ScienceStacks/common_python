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


###################################################
class FeatureHandler(object):
  """
  Default handler for selecting features for a class.
  Provides ordered values of features.
  """

  def __init__(self, df_X, ser_y):
    """
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: class
    """
    self.df_X = df_X
    self.ser_y = ser_y
    self.all_features = df_X.columns.tolist()
    self._feature_dct = None

  @property
  def feature_dct(self):
    """
    Features ordered in descending priority for each class.
    :return dict:
        key: class
        value: list of features in descending order
    """
    if self._feature_dct is None:
      df_fstat = util_classifier.makeFstatDF(df_X, ser_y)
      for cls in ser_y.unique():
        ser_fstat = df_fstat[cls]
        ser_fstat.sort_values()
        self._feature_dct[cls] = ser_fstat.index.tolist()
    return self._feature_dct

  def getFeatures(self, cls):
    return self.feature_dct[cls]

  def getNonFeatures(self, features):
    return list(set(self.all_features).difference(
        features))


class MultiClassifier(object):
 
  def __init__(self, base_clf=svm.LinearSVC(),
        feature_handler=FeatureHandler(),
        desired_accuracy=0.9):
    """
    :param Classifier base_clf: 
    :param FeatureHandler feature_handler:
    :param float desired_accuracy: accuracy for clf
    """
    self.base_clf = base_clf
    self.classes = []
    self.all_features = None
    self.feature_handler = None
    self.feature_dct = {}  # Features used for classifier
    self.clf_dct = {}  # key is cls; value is clf
    self.score_dct = {}  # key is cls; value is score

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
        assigned: all_features, classes, features_policy, 
                  clf_dct, score_dct, feature_dct
       
    """
    self.feature_handler = FeatureHandler(df_X, ser_y)
    self.classes = ser_y.unique()
    for cls in self.classes:
      # Initialize for this class
      self.clf_dct[cls] = copy.deepcopy(self.base_clf)
      features_cls = self.feature_handler.getFeatures(cls)
      ser_y_cls = util_classifier.makeOneStateSer(ser_y,
          cls)
      # Use enough features to obtain the desired accuracy
      # This may not be possible
      for rank in range(len(features)):
        self.feature_dct[cls] = features_cls[0:rank+1]
        df_X_rank = self._setFeatures(df_X, cls)
        self.clf_dct[cls].fit(df_X_rank, ser_y_rank)
        self.score_dct[cls] = clf.score(df_X, ser_y)
        if self.score_dct[cls] >= self.desired_accuracy:
          break

  def _setFeatures(self, cls):
    """
    Initializes the features for a class.
    :return pd.DataFrame: Non-feature columns are 0
    :globals:
        read: all_features
    """
    non_features = self.feature_handler.getNonFeatures(
        self.feature_dct[cls])
    df_X_sub = self.feature_handler.df_X.copy()
    df_X_sub[non_features] = 0
    return df_X_sub

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
    for cls in self.classes:
      df_X_sub = self._setFeatures(df_X, cls)
      df_pred[cls] = self.clf_dct[cls].predict(df_X_sub)
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
    for cls in self.classes:
      sel = ser_y == cls
      df_predict_sub = df_predict.loc[sel, :]
      score += df_predict_sub[cls].sum()
      total += len(df_pred_sub)
    if total != len(df_X):
      raise RuntimeError("Should process each row once.")
    return score/len(df_X)
