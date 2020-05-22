'''Calculates various statistics for feature selection'''

"""
Single feature accuracy (SFA). Accuracy of a classifier
using a single feature. We denote the accuracy
of a classifier with the single feature F by A(F).

Classifier prediction correlation (CPC). Correlation of
the predictions of two single feature classifiers.
Let P(F) be the predictions produced by a classifier
with just the feature F. Then CPC(F1, F2) is
corr(P(F1), P(F2)).

Incremental prediction accuracy (IPA). Increase in
classification accuracy by using two features in
combination instead of the most accurate of a
two feature classifier.
IPA(F1, F2) = A(F1, F2) - max(A(F1), A(F2))
"""

import common_python.constants as cn
from common_python.classifier import util_classifier

import copy
import numpy as np
import os
import pandas as pd
from sklearn import svm


NUM_CROSS_VALID = 50  # Number of cross validations
FEATURE1 = "feature1"
FEATURE2 = "feature2"


class FeatureAnalyzer(object):

  def __init__(self, clf, df_X, ser_y,
      num_cross_valid=NUM_CROSS_VALID):
    """
    :param Classifier clf: binary classifier
    :param pd.DataFrame df_X:
        columns: features
        rows: instances
    :param pd.Series ser_y:
        values: 0, 1
        rows: instances
    """
    ######### PRIVATE ################
    self._clf = copy.deepcopy(clf)
    self._df_X = df_X
    self._ser_y = ser_y
    self._features = df_X.columns.tolist()
    self._partitions = [
        util_classifier.partitionByState(self._ser_y)
        for _ in range(num_cross_valid)]
    # Single feature accuracy
    self._ser_sfa = None
    # classifier prediction correlation
    self._df_cpc = None
    # incremental prediction accuracy
    self._df_ipa = None

  @property
  def ser_sfa(self):
    if self._ser_sfa is None:
      scores = []
      for feature in self._features:
        df_X = pd.DataFrame(
            self._clf, self._df_X[feature])
        score = util_classifier.binaryCrossValidate(
            self._clf, df_X, self._ser_y,
            partitions=self._partitions)
        scores.append(score)
      self._ser_y = pd.Series(
          scores, index=self._features)
    return self._ser_y

  @property
  def df_cpc(self):
    """
    creates pd.DataFrame
        row index: features
        columns: features
        scores: correlation
    """
    if self._df_cpc is None:
      dct = {FEATURE1: [], FEATURE2{}, cn.SCORE: []}
      for feature1 in self._features:
        for feature2 in self._features:
          clf_desc1 =  \
              util_classifier.ClassifierDescription(
              clf=self._clf, features=[feature1])
          clf_desc2 =  \
              util_classifier.ClassifierDescription(
              clf=self._clf, features=[feature2])
          score = util_classifier.correlatePredictions(
              clf_desc1, clf_desc2, self._df_X,
              self._ser_y, self._partitions)
          dct[FEATURE1].append(feature1)
          dct[FEATURE2].append(feature2)
          dct[cn.SCORE].append(score)
      df = pd.DataFrame(dct)
      self._df_cpc = pd.pivot_table(index=FEATURE1,
          columns=FEATURE2, values=cn.SCORE)
    return self._df_cpc

  @property
  def df_ipa(self):
    """
    creates pd.DataFrame
        row index: features
        columns: features
        scores: incremental accuracy
    """
    def predict(features):
      df_X = pd.DataFrame(self._df_X[feature])
      return util_classifier.binaryCrossValidate(
          self._clf, df_X, self._ser_y,
          partitions=self._partitions)
    #
    if self._df_ipa is None:
      dct = {FEATURE1: [], FEATURE2{}, cn.SCORE: []}
      for feature1 in self._features:
        for feature2 in self._features:
          score1 = predict([feature1])
          score2 = predict([feature2])
          score3 = predict([feature1, feature2])
          score = score3 - max(score1, score2)
          dct[FEATURE1].append(feature1)
          dct[FEATURE2].append(feature2)
          dct[cn.SCORE].append(score)
      df = pd.DataFrame(dct)
      self._df_ipa = pd.pivot_table(index=FEATURE1,
          columns=FEATURE2, values=cn.SCORE)
    return self._df_ipa
