"""
Implements meta-classifiers, classifiers to handle multiple
sets of features.
 1. Constructor signature: clf - a classifier
 2. Fit signature: list-df_feature, ser_label
 3. Also implements: predict, score
"""

import copy
import numpy as np
import pandas as pd


class MetaClassifier(object):
  # Abstract class. Must subclass and implement fit method.

  def __init__(self, clf):
    """
    :param Classifier clf: implements fit, predict, score
    """
    self.clf = clf
    self._is_fit = False
    self.majority_class = None

  def fitMajorityClass(self, _, ser_label, **__):
    """
    Fit for a majority class classifier.
    :param pd.Series df_class:
        index: instance
        value: class label
    Updates
      self.majority_class - most frequnetly occurring class
    """
    ser = ser_label.value_counts()
    self.majority_class = ser[ser.index[0]]

  def _check(self):
    if not self._is_fit:
      raise ValueError("Must do fit before attempting predict or score.")

  def fit(self, dfs_feature, ser_label, **kwargs):
    """
    :param list-pd.DataFrame df_feature: each dataframe
        index: instance
        column: feature
    :param pd.Series df_class:
        index: instance
        value: class label
    :param dict kwargs: optional arguments if any
    Notes
      1. len(df_feature) == len(ser_label)
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
    self.clf.predict(df_feature, **kwargs)

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

  def scoreMajorityClass(self, _, ser_label, **__):
    """
    Scores for a majority class classifier
    :param pd.Series ser_label:
        index: instance
        value: class label
    :return float:
    """
    self._check()
    num_match = [c == self.majority_class for c in ser_label]
    return num_match / len(ser_label)

  def score(self, dfs_feature, ser_label, is_relative=False, **kwargs):
    """
    Scores a previously fitted classifier.
    :param list-pd.DataFrame df_feature: each dataframe
        index: instance
        column: feature
    :param pd.Series ser_label:
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
        zip(ser_label.values, ser_predicted.values)]
    score_raw = num_match / len(ser_predicted)
    if is_relative:
      score_majority_class = self.scoreMajorityClass(dfs_feature,
           ser_label, **kwargs)
      score = (score_raw - score_majority_class) / (1 - score_majority_class)
    else:
      score = score_raw
    return score


class MetaClassifierDefault(MetaClassifier):
  # Trains on the first instance of feature data

  def fit(self, dfs_feature, ser_label, **kwargs):
    df_feature = dfs_feature[0]
    self.clf.fit(df_feature, ser_label, **kwargs)
    self._is_fit = True


class MetaClassifierAverage(MetaClassifier):
  # Trains on the average of feature values.

  def fit(self, dfs_feature, ser_label, **kwargs):
    """
    Does fit on the average of the feature values.
    """
    df_feature = dfs_feature[0].copy()
    for df in dfs_features[1:]:
      df_feature += df
    df_feature = df_feature.applymap(lambda v: v / len(dfs_feature))
    #
    self.clf.fit(df_feature, ser_label, **kwargs)
    self._is_fit = True


class MetaClassifierAugmentInstances(MetaClassifier):
  # Uses replicas as additional instances for training.

  def fit(self, dfs_feature, ser_label, **kwargs):
    """
    Includes replications as separate instances
    """
    df_feature = pd.concat(dfs_feature)
    sers_class = [ser_label for _ in range(len(dfs_feature))]
    ser_label_replicas = pd.concat(sers_class)
    self.clf.fit(df_feature, ser_label_replicas, **kwargs)
    self._is_fit = True
