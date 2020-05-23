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


NUM_CROSS_ITER = 50  # Number of cross validations
FEATURE1 = "feature1"
FEATURE2 = "feature2"
SFA = "sfa"
CPC = "cpc"
IPA = "ipa"
METRICS = [SFA, CPC, IPA]


class FeatureAnalyzer(object):

  def __init__(self, clf, df_X, ser_y,
      num_cross_iter=NUM_CROSS_ITER, is_report=True,
      report_interval=None):
    """
    :param Classifier clf: binary classifier
    :param pd.DataFrame df_X:
        columns: features
        rows: instances
    :param pd.Series ser_y:
        values: 0, 1
        rows: instances
    :parm int num_cross_iter: iterations in cross valid
    :param int report_interval: number processed
    """
    ######### PRIVATE ################
    self._clf = copy.deepcopy(clf)
    self._df_X = df_X
    self._ser_y = ser_y
    self._is_report = is_report
    self._features = df_X.columns.tolist()
    self._partitions = [
        util_classifier.partitionByState(self._ser_y)
        for _ in range(num_cross_iter)]
    self._report_interval = report_interval
    # Number procesed since last report
    self._num_processed = 0
    # Single feature accuracy
    self._ser_sfa = None
    # classifier prediction correlation
    self._df_cpc = None
    # incremental prediction accuracy
    self._df_ipa = None

  def _reportProgress(self, metric, count, total):
    """
    Reports progress on calculation.
    :param str metric:
    :param int count: how much completed
    :param int total: how much total
    """
    if self._report_interval is not None:
      if count == 0:
        self._num_processed = 0
      elif count - self._num_processed >=  \
          self._report_interval:
        self._num_processed = count
        if self._is_report:
          print("\n***Progress for %s: %d/%d" %
              (metric, count, total))
      else:
        pass

  @property
  def ser_sfa(self):
    """
    Construct series for single feature accuracy
      index: feature
      value: accuracy in [0, 1]
    """
    total = len(self._features)
    if self._ser_sfa is None:
      self._num_processed = 0
      scores = []
      for feature in self._features:
        df_X = pd.DataFrame(self._df_X[feature])
        score = util_classifier.binaryCrossValidate(
            self._clf, df_X, self._ser_y,
            partitions=self._partitions)
        scores.append(score)
        self._reportProgress(SFA, len(scores), total)
      self._ser_sfa = pd.Series(
          scores, index=self._features)
    return self._ser_sfa

  @property
  def df_cpc(self):
    """
    creates classifier predicition correation pd.DataFrame
        row index: features
        columns: features
        scores: correlation
    """
    total = (len(self._features))**2
    dct = {FEATURE1: [], FEATURE2: [], cn.SCORE: []}
    if self._df_cpc is None:
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
          self._reportProgress(CPC,
              len(dct[FEATURE1]), total)
      df = pd.DataFrame(dct)
      self._df_cpc = df.pivot(index=FEATURE1,
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
      df_X = pd.DataFrame(self._df_X[features])
      return util_classifier.binaryCrossValidate(
          self._clf, df_X, self._ser_y,
          partitions=self._partitions)
    #
    total = (len(self._features))**2
    dct = {FEATURE1: [], FEATURE2: [], cn.SCORE: []}
    if self._df_ipa is None:
      for feature1 in self._features:
        for feature2 in self._features:
          score1 = predict([feature1])
          score2 = predict([feature2])
          score3 = predict([feature1, feature2])
          score = score3 - max(score1, score2)
          dct[FEATURE1].append(feature1)
          dct[FEATURE2].append(feature2)
          dct[cn.SCORE].append(score)
          self._reportProgress(CPC,
              len(dct[FEATURE1]), total)
      df = pd.DataFrame(dct)
      self._df_ipa = df.pivot(index=FEATURE1,
          columns=FEATURE2, values=cn.SCORE)
    return self._df_ipa
