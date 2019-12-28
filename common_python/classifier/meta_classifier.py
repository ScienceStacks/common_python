"""
Implements meta-classifiers, classifiers to handle multiple
replications of feature values for the same instance.
Each meta-classifier implements _makeTrainingData.
"""

from common_python.classifier.majority_class import MajorityClass

import copy
import numpy as np
import pandas as pd
from sklearn import svm, model_selection


##########################################
class MetaClassifier(object):
  # Abstract class. Must subclass and implement _makeTrainingData method.

  def __init__(self, clf=svm.LinearSVC()):
    """
    :param Classifier clf: implements fit, predict, score
    """
    self.clf = clf
    self._is_fit = False
    self.majority_clf = MajorityClass()

  def _check(self):
    if not self._is_fit:
      raise ValueError("Must do fit before attempting predict or score.")

  def _makeTrainingData(self, dfs_feature, ser_label):
    """
    Creates the training data from replicated features. This is done
    in different ways by the subclasses and so this method must be
    overridden.
    :param list-pd.DataFrame df_feature: each dataframe
        index: instance
        column: feature
    :param pd.Series df_class:
        index: instance
        value: class label
    """
    raise RuntimeError("Must override")

  def fit(self, dfs_feature, ser_label):
    """
    :param list-pd.DataFrame df_feature: each dataframe
        index: instance
        column: feature
    :param pd.Series df_class:
        index: instance
        value: class label
    Notes
      1. len(df_feature) == len(ser_label)
    Updates
      self.clf
    """
    df_feature, ser_label = self._makeTrainingData(
        dfs_feature, ser_label)
    self.clf.fit(df_feature, ser_label)
    # Fit for the majority class
    self.majority_clf.fit(_, ser_label)
    #
    self._is_fit = True

  def predict(self, df_feature):
    """
    :param pd.DataFrame df_feature:
        index: instance
        column: feature
    :return pd.Series: 
        index: instance
        value: predicted class label
    """
    self._check()
    return pd.Series(self.clf.predict(df_feature))

  def crossValidate(self, dfs_feature, ser_label, **kwargs):
    """
    Does cross validation for the classifier.
    :param dict kwargs: Options for cross validation
    :return float, float: mean, std of accuracy
    """
    df_feature, ser_label = self._makeTrainingData(dfs_feature, ser_label)
    cv_result = model_selection.cross_validate(
        self.clf, df_feature, ser_label, **kwargs)
    return np.mean(cv_result['test_score']), np.std(cv_result['test_score'])

  def score(self, df_feature, ser_label):
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
    :return float, score: absolute accuracy, relative accuracy
    """
    self._check()
    ser_predicted = self.predict([df_feature])
    num_match = [c1 == c2 for c1, c2 in 
        zip(ser_label.values, ser_predicted.values)]
    score_raw = num_match / len(ser_predicted)
    score_majority = self.majority_class.predict(_, ser_label)
    score_relative = (score_raw - score_majority) / (1 - score_majority)
    return score_raw, score_relative



##########################################
class MetaClassifierDefault(MetaClassifier):
  # Trains on the first instance of feature data

  def _makeTrainingData(self, dfs_feature, ser_label):
    return dfs_feature[0], ser_label


##########################################
class MetaClassifierAverage(MetaClassifier):
  # Trains on the average of feature values.

  def _makeTrainingData(self, dfs_feature, ser_label):
    """
    Does fit on the average of the feature values.
    """
    df_feature = dfs_feature[0].copy()
    for df in dfs_features[1:]:
      df_feature += df
    df_feature = df_feature.applymap(lambda v: v / len(dfs_feature))
    return df_feature, ser_label


##########################################
class MetaClassifierAugmentInstances(MetaClassifier):
  # Uses replicas as additional instances for training.

  def _makeTrainingData(self, dfs_feature, ser_label):
    df_feature = pd.concat(dfs_feature)
    sers_class = [ser_label for _ in range(len(dfs_feature))]
    ser_label_replicas = pd.concat(sers_class)
    return df_feature, ser_label_replicas
