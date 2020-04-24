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
from common_python.classifier import feature_selector
from common_python.util.persister import Persister

import copy
import numpy as np
import os
import pandas as pd
import random
from sklearn import svm

## Hyperparameters for classifier
# The following control the stopping criteria
# in forward selection. The first criteria
# that is satisfied halts forward selection.
# 
MAX_ITER = 20  # Maximum number of features considered
MAX_DEGRADE = 0.05 # Maximum degradation from best score
#  The following controls acceptance of a feature
MIN_INCR_SCORE = 0.02  # Minimum amount by which score
                       # Score must increase to be included

# Files
# Serialize results
DIR = os.path.dirname("__file__")
SERIALIZE_PATH = os.path.join(DIR, "multi_classifier.pcl")
PERSISTER_INTERVAL = 5


class MultiClassifier(object):
  """
  Has several hyperparameters defined in constructor.
  """
 
  def __init__(self, base_clf=svm.LinearSVC(),
        feature_selector_cls=  \
        feature_selector.FeatureSelectorResidual,
        min_incr_score=MIN_INCR_SCORE,
        max_iter=MAX_ITER, max_degrade=MAX_DEGRADE,
        **kwargs):
    """
    :param Classifier base_clf: 
    :param FeatureSelector feature_selector_cls:
    :param float min_incr_score: min amount by which
        a feature must increase the score to be included
    :param int max_iter: maximum number of iterations
    :param float max_degrade: maximum difference between
        best score and actual
    :param dict kwargs: arguments passed to FeatureSelector
    """
    # Public
    self.classes = []
    self.clf_dct = {}  # key is cls; value is clf
    self.score_dct = {}  # key is cls; value is score
    self.best_score_dct = {}  # key is cls; value is score
    self.selector = None  # Constructed during fit
    # Private
    self._min_incr_score = min_incr_score
    self._max_degrade = max_degrade
    self._max_iter = max_iter
    self._kwargs = kwargs
    self._feature_selector_cls = feature_selector_cls
    self._base_clf = base_clf

  def fit(self, df_X, ser_y, persister=None):
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
    :param Persister persister: used to save state
    ASSIGNED INSTANCE VARIABLES
        _classes, clf_dct, score_dct
    """
    def makeArrays(df_X, df_y):
      arr_X = df_X.values[indices_score, :]
      arr_y = self.ser_y_cls.values[indices_score]
      return arr_X, arr_y
    #
    self.classes = ser_y.unique()
    finished_dct = {c: False for c in self.classes}
    initialized_dct = {c: False for c in self.classes}
    # Handle restart of a fit, detecting which have
    if self.selector is not None:
      # Previously run fit
      for cls in self.classes:
        if cls in self.clf_dct:
          pos = list(self.classes).index(cls)
          if pos > 0:
            # Completed the previous class
            finished_dct[self.classes[pos - 1]] = True
          if cls in self.clf_dct.keys():
            initialized_dct[cls] = True
    else:
      self.selector = self._feature_selector_cls(
          df_X, ser_y, **self._kwargs)
    # Do fit by class
    for cls in self.classes:
      if finished_dct[cls]:
        continue
      if not initialized_dct[cls]:
        # Initialize for this class
        clf = copy.deepcopy(self._base_clf)
        self.clf_dct[cls] = clf
        self.ser_y_cls = util_classifier.makeOneStateSer(ser_y,
            cls)
      else:
        clf = self.clf_dct[cls]
      last_score = 0
      # Select the indices to be used for prediction
      length = len(ser_y)
      indices_cls = [i for i, v in enumerate(self.ser_y_cls)
          if v == 1]
      indices_non_cls = set(
          range(length)).difference(indices_cls)
      indices_score = random.sample(indices_non_cls,
          len(indices_non_cls))
      indices_score.extend(indices_cls)
      # Find maximum accuracy achievable for this class
      clf.fit(df_X, self.ser_y_cls)  # Fit for all features
      arr_X, arr_y = makeArrays(df_X, self.ser_y_cls)
      self.best_score_dct[cls] = clf.score(arr_X, arr_y)
      # Use enough features to obtain the desired accuracy
      # This may not be possible
      length = len(self.selector.all_features)
      num_iter = len(self.selector.feature_dct[cls])  \
          + len(self.selector.remove_dct[cls])
      for rank in range(num_iter, length):
        if num_iter % PERSISTER_INTERVAL == 0:
          if persister is not None:
            persister.set(self)
        num_iter += 1
        if num_iter > self._max_iter:
          break
        if not self.selector.add(cls):
          break
        df_X_rank = self.selector.zeroValues(cls)
        self.clf_dct[cls].fit(df_X_rank, self.ser_y_cls)
        arr_X, arr_y = makeArrays(df_X_rank, self.ser_y_cls)
        new_score = self.clf_dct[cls].score(arr_X, arr_y)
        if new_score - last_score > self._min_incr_score:
            self.score_dct[cls] = new_score
            last_score = new_score
        else:
          # Remove the feature
          self.selector.remove(cls)
        if self.best_score_dct[cls]  \
            - self.score_dct[cls]  < self._max_degrade:
          break

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
      df_pred[cls] = self.clf_dct[cls].predict(df_X)
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
      sel = ser_y == cls
      df_predict_sub = df_predict.loc[sel, :]
      score += df_predict_sub[cls].sum()
      total += len(df_pred_sub)
    if total != len(df_X):
      raise RuntimeError("Should process each row once.")
    return score/len(df_X)

  @classmethod
  def doQualityFit(cls, df_X, ser_y,
       max_iter=5000,
       path=SERIALIZE_PATH):
    """
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: class
    :param str path: serialization file
    :param int max_iter: used for tests
    :return MultiClassifier:
    """
    persister = Persister(path)
    if persister.isExist():
      print ("Previous state found.")
      multi = persister.get()
    else:
      print ("No previous state found.")
      multi = MultiClassifier(feature_selector_cls=  \
          feature_selector.FeatureSelector,
          max_iter=max_iter, max_degrade=0.01)
    multi.fit(df_X, ser_y, persister=persister)
