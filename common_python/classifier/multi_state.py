"""
Classifier for multiple states constructed from a
base classifier. Classifier features
are selected separately for each state (class).

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
import pandas as pd
from sklearn import svm


###################################################
class MultiState(object):
 
  def __init__(self, base_clf=svm.LinearSVC(),
        desired_accuracy=0.9):
    """
    :param Classifier base_clf: 
    :param float desired_accuracy: accuracy for clf
    """
    self.base_clf = base_clf
    self.states = []
    self.all_features = None
    self.feature_dct = {}  # key is state; value is list
    self.clf_dct = {}  # key is state; value is clf
    self.score_dct = {}  # key is state; value is score

  def fit(self, df_X, ser_y):
    """
    Selects the top features for each state and fits
    a classifier with the desired accuracy.
    Classifiers are in self.clfs.
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: state (class)
    :globals:
        assigned: all_features, states, features_dct, 
                  clf_dct, score_dct
       
    """
    self.states = ser_y.unique()
    self.all_features = df_X.columns.tolist()
    df_fstat = util_classifier.makeFstatDF(df_X, ser_y)
    for state in self.states:
      # Initialize for this state
      self.clf_dct[state] = copy.deepcopy(self.base_clf)
      ser_fstat = df_fstat[state]
      ser_fstat.sort_values()
      features_state = ser_fstat.index.tolist()
      ser_y_state = util_classifier.makeOneStateSer(ser_y,
          state)
      # Use enough features to obtain the desired accuracy
      # This may not be possible
      for rank in range(len(features)):
        self.feature_dct[state] = features_state[0:rank+1]
        df_X_rank = self._setFeatures(df_X, state)
        self.clf_dct[state].fit(df_X_rank, ser_y_rank)
        self.score_dct[state] = clf.score(df_X, ser_y)
        if self.score_dct[state] >= self.desired_accuracy:
          break

  def _setFeatures(df_X, state):
    """
    Initializes the features for a state.
    :return pd.DataFrame: Non-feature columns are 0
    :globals:
        read: all_features
    """
    non_features = list(set(
        self.all_features).difference(
        self.features_dct[state]))
    df_X_sub = df_X.copy()
    df_X_sub[non_features] = 0
    return df_X_sub

  def predict(self, df_X):
    """
    Predict using the results of the classifier for each
    state. Prediction probability is the fraction of states
    that predict a 1 for the instance.
    :param pd.DataFrame: features, indexed by instance.
    :return pd.DataFrame: fraction prediction by state;
        column: states
        index: instance
        values: fraction
    :globals:
        read: clf_dct
    """
    df_pred = pd.DataFrame()
    for state in self.states:
      df_X_sub = self._setFeatures(df_X, state)
      df_pred[state] = self.clf_dct[state].predict(df_X_sub)
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
        values: state
    """
    score = 0
    total = 0
    df_predict = self.predict(df_X)
    for state in self.states:
      sel = ser_y == state
      df_predict_sub = df_predict.loc[sel, :]
      score += df_predict_sub[state].sum()
      total += len(df_pred_sub)
    if total != len(df_X):
      raise RuntimeError("Should process each row once.")
    return score/len(df_X)
