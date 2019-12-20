"""
Implements policies for handling noise in features.

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
  # Abstract class

  def __init__(self, clf):
    """
    :param Classifier clf: implements fit, predict, score
    """
    self.clf = clf

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
    return self.clf.predict(df_feature, **kwargs)

  def score(self, dfs_feature, ser_class, **kwargs):
    """
    Scores a previously fitted classifier.
    :param list-pd.DataFrame df_feature: each dataframe
        index: instance
        column: feature
    :param pd.Series ser_class:
        index: instance
        value: class label
    :param dict kwargs: optional arguments if any
    :return float:
    """
    ser_predicted = self.predict(dfs_features)
    num_match = [c1 == c2 for c1, c2 in 
        zip(ser_class.values, ser_predicted.values)]
    return num_match / len(ser_predicted)


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
    Does fit on the average of the feature values.
    """
    df_feature = pd.concat(dfs_feature)
    sers_class = [ser_class for _ in range(len(dfs_feature))]
    ser_class_replicas = pd.concat(sers_class)
    self.clf.fit(df_feature, ser_class_replicas, **kwargs)
