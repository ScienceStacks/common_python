"""
Implements meta-classifier with policies for handling feature noise.

Case 1: Instances are replicated.
  a. Average instances
  b. Use replications as additional features

Case 2: Thresholds


Case 3: Thresholds with replicas.
"""

import copy
import numpy as np
import pandas as pd


class FeatureNoiseClassifier(object):
  # Abstract class. Must subclass and implement fit method.

  def __init__(self, clf):
    """
    :param Classifier clf: implements fit, predict, score
    """
    self.clf = clf
    self.majority_class = None

  def fitMajorityClass(self, _, ser_class, **__):
    """
    Fit for a majority class classifier.
    :param pd.Series df_class:
        index: instance
        value: class label
    Updates
      self.majority_class - most frequnetly occurring class
    """
    ser = ser_class.value_counts()
    self.majority_class = ser[ser.index[0]]

  def _check(self):
    if self.majority_class is None:
      raise ValueError("Must do fit before attempting predict or score.")

  def fit(self, dfs_feature, ser_class, **kwargs):
    """
    :param list-pd.DataFrame df_feature: each dataframe
        index: instance
        column: feature
    :param pd.Series df_class:
        index: instance
        value: class label
    :param dict kwargs: optional arguments if any
    Updates
      self.clf
      self.majority_class - most frequnetly occurring class
    """
    raise RuntimeError("Must override")

  def predict(self, df_feature, **kwargs):
    """
    :param pd.DataFrame df_feature:
        index: instance
        column: feature
    :param dict kwargs: optional arguments if any
    :return pd.Series: 
        index: instance
        value: predicted class label
    """
    self._check()
    return self.clf.predict(df_feature, **kwargs)

  def predictMajorityClass(self, df_feature, **kwargs):
    """
    :param pd.DataFrame df_feature:
        index: instance
        column: feature
    :param dict kwargs: optional arguments if any
    :return pd.Series: 
        index: instance
        value: predicted class label
    """
    self._check()
    return pd.Series(np.repeat(self.majority_class, len(df_feature)))

  def scoreMajorityClass(self, _, ser_class, **__):
    """
    Scores for a majority class classifier
    :param pd.Series ser_class:
        index: instance
        value: class label
    :return float:
    """
    self._check()
    num_match = [c == self.majority_class for c in ser_class]
    return num_match / len(ser_class)

  def score(self, dfs_feature, ser_class, is_relative=False, **kwargs):
    """
    Scores a previously fitted classifier.
    :param list-pd.DataFrame df_feature: each dataframe
        index: instance
        column: feature
    :param pd.Series ser_class:
        index: instance
        value: class label
    :param bool is_relative: score is relative to majority class classifier
        1: perfect classification
        0: equivalent to majority class classifier
        neg: worse than majority class classifier
    :param dict kwargs: optional arguments if any
    :return float:
    """
    self._check()
    ser_predicted = self.predict(dfs_features)
    num_match = [c1 == c2 for c1, c2 in 
        zip(ser_class.values, ser_predicted.values)]
    score_raw = num_match / len(ser_predicted)
    if is_relative:
      score_majority_class = self.scoreMajorityClass(dfs_feature,
           ser_class, **kwargs)
      score = (score_raw - score_majority_class) / (1 - score_majority_class)
    else:
      score = score_raw
    return score


class FeatureNoiseClassifierAverage(FeatureNoiseClassifier):
  # Uses average of the feature values

  def fit(self, dfs_feature, ser_class, **kwargs):
    """
    Does fit on the average of the feature values.
    """
    df_feature = dfs_feature[0].copy()
    for df in dfs_features[1:]:
      df_feature += df
    df_feature = df_feature.applymap(lambda v: v / len(dfs_feature))
    #
    self.clf.fit(df_feature, ser_class, **kwargs)


class FeatureNoiseClassifierReplicas(FeatureNoiseClassifier):
  # Uses average of the feature values

  def fit(self, dfs_feature, ser_class, **kwargs):
    """
    Includes replications as separate instances
    """
    df_feature = pd.concat(dfs_feature)
    sers_class = [ser_class for _ in range(len(dfs_feature))]
    ser_class_replicas = pd.concat(sers_class)
    self.clf.fit(df_feature, ser_class_replicas, **kwargs)
