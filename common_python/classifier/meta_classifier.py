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


# Result of scoring a classifier
#  abs: absolute accuracy in [0, 1] - fraction correct
#  rel: relative accuracy in [-inf, 1] - fraction
#     of instances classified incorrectly by PluralityClassifier
#     that are correctly classified by the candidate classifier
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

  def _calcMatch(self, df_feature, ser_label):
    """
    Calculates the number of matches of instances.
    :param pd.DataFrame df_feature:
    :param pd.Series ser_label:
    :return float:
    """
    ser_predicted = self.predict(df_feature)
    return sum([1 for c1, c2 in 
        zip(ser_label.values, ser_predicted.values) if c1 == c2])

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
    num_match = self._calcMatch(df_feature, ser_label)
    score_abs = num_match / len(ser_label)
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


##########################################
class MetaClassifierEnsemble(MetaClassifier):
  # Create an ensemble of classifier from each feature replica

  def __init__(self, clf):
    super.__init__(clf=clf)
    self.ensemble = None

  def fit(self, dfs_feature, ser_label):
    self.ensemble = []
    for df in dfs_feature:
      clf = copy.deepcopy(self.clf)
      self.ensemble.append(clf.fit(df, ser_label))
    self._is_fit = True

  def predict(self, df_feature):
    """
    Does an ensemble prediction in which the most frequently
    predicted label is the ensemble prediction with
    random selection among ties.
    :param pd.DataFrame df_feature:
        columns: features
        rows: instances
    :return pd.Series:
        values: labels
        rows: instances
    """
    def selectLabel(ser):
       """
       Selects the most frequently occurring label and randomly
       select among ties.
       """
       ser_count = ser.value_counts()
       ser_top = ser_count[ser_count==ser_count[ser_count.index[0]]]
       idx = np.random.randint(0, len(ser_top))
       return ser_top.index[idx]
    #
    self._check()
    # DataFrame where each row is a classifier prediction
    df_label = pd.concat(
        [clf.predict(df_feature) for clf in self.ensemble],
        axis=1)
    df_label = df_label.T
    # Predictions
    predicts = [selectLabel(df_label[c]) for c in df_label.columns]
    #
    return pd.Series(predicts)

  def _calcMatch(self, df_feature, ser_label, iter_int=10):
    """
    Calculates the number of matches of instances in score.
    This is done repeatedly for ensemble to handle ties.
    :param pd.DataFrame df_feature:
    :param pd.Series ser_label:
    :param int iter_int: number of iterations in the matching
    :return float:
    """
    matches = []
    for _ in range(iter_int):
      ser_predicted = self.predict(df_feature)
      num_match = sum([1 for c1, c2 in 
          zip(ser_label.values, ser_predicted.values) if c1 == c2])
      matches.append(num_match)
    return np.mean(matches)
