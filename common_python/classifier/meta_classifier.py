"""
Implements meta-classifiers, classifiers to handle multiple
replications of feature values for the same instance.
Each meta-classifier implements _makeTrainingData.
"""

from common_python.classifier.plurality_classifier  \
    import PluralityClassifier

import collections
import copy
import numpy as np
import pandas as pd
from sklearn import svm, model_selection


ScoreResult = collections.namedtuple("ScoreResult",
    "abs rel")


##########################################
class MetaClassifier(object):
  # Abstract class. Must subclass and implement _makeTrainingData method.

  def __init__(self, clf=svm.LinearSVC()):
    """
    :param Classifier clf: implements fit, predict, score
    """
    self.clf = clf
    self._is_fit = False
    self.plurality_clf = PluralityClassifier()

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
    :return ScoreResult:
    """
    self._check()
    self.plurality_clf.fit(df_feature, ser_label)
    ser_predicted = self.predict(df_feature)
    num_match = sum([1 for c1, c2 in 
        zip(ser_label.values, ser_predicted.values) if c1 == c2])
    score_abs = num_match / len(ser_predicted)
    score_plurality = self.plurality_clf.score(df_feature, ser_label)
    score_rel = (score_abs - score_plurality) / (1 - score_plurality)
    return ScoreResult(abs=score_abs, rel=score_rel)


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
class MetaClassifierAugment(MetaClassifier):
  # Uses replicas as additional instances for training.

  def _makeTrainingData(self, dfs_feature, ser_label):
    df_feature_replica = pd.concat(dfs_feature)
    sers = [ser_label for _ in range(len(dfs_feature))]
    ser_label_replica = pd.concat(sers)
    return df_feature_replica, ser_label_replica


##########################################
class MetaClassifierPlurality(MetaClassifier):
  # Uses wrapper for plurality classifier

  def __init__(self):
    super.__init__(clf=PluralityClassifier())

  def _makeTrainingData(self, _, ser_label):
    return None, ser_label
