"""
Classifier for data with multiple classes using
a base classifier that is copied to instantiate
a separate classifier for each class.
The classifier is provided with features for
each class.

The base classifier must expose methods for fit,
predict, score. Fitting is done by forward selection
of features to achieve a desired accuracy.
This is a "soft" constraint, and it is checked
without the use of cross validation (for performance
reasons).
"""

from common_python.classifier import util_classifier

import copy
import numpy as np
import os
import pandas as pd
import random
from sklearn import svm


class MultiClassifier(object):
 
  def __init__(self, base_clf=svm.LinearSVC(),
        feature_dct=None):
    """
    :param Classifier base_clf: 
    :param dict feature_dct: Features for each class
    """
    ###### PRIVATE ##### 
    self._feature_dct = feature_dct
    self._base_clf = base_clf
    ###### PUBLIC ##### 
    # key is cls; value is clf
    self.classes = None
    self.clf_dct = {}
    self.score_dct = {}  # key is cls; value is score
    self.best_score_dct = {}  # key is cls; value is score

  def _makeXArray(self, cls, df_X):
    return df_X[self._features_dct[cls]].values

  def _makeYArray(self, cls, ser_y):
    ser_y_1 = util_classifiermakeOneStateSer(ser_y, cls)
    return ser_y_1.values

  def fit(self, df_X, ser_y):
    """
    Selects the top features for each class and fits
    a classifier with the desired accuracy by including
    more features as selected by FeatureCollection.
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: class
    """
    self.classes = ser_y.unique()
    columns = df_X.columns.tolist()
    if self._feature_dct is None:
      # By default, all columns are used for the features of a class.
      self._feature_dct = {c: columns for c in self.classes}
    # Do fit for each class using class specific features
    for cls in self.classes:
      self.clf_dct[cls] = copy.deepcopy(self._base_clf)
      arr_X = self._makeXArray(cls, df_X)
      arr_y = self._makeYArray(cls, ser_y)
      self.clf_dct[cls].fit(arr_X, arr_y)

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
    for cls in self.classes:
      arr_X = self._makeXArray(cls, df_X)
      df_pred[cls] = self.clf_dct[cls].predict(arr_X)
    df_pred.index = df_X.index
    ser_tot = df_pred.sum(axis=1)
    for cls in self.classes:
      df_pred[cls] = df_pred[cls]/ser_tot
    df_pred = df_pred.fillna(0)
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
      # Select instances for this class
      sel = ser_y == cls
      df_predict_sub = df_predict.loc[sel, :]
      # Sum the predictions for the correct class
      score += df_predict_sub[cls].sum()
      total += len(df_pred_sub)
    if total != len(df_X):
      raise RuntimeError("Should process each row once.")
    return score/len(df_X)
