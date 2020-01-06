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

MAX_ITER = 500  # Maximum number of iterations for score calculation
SVM_KWARGS = {"max_iter": 50000}


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

  def __init__(self, clf=None):
    """
    :param Classifier clf: implements fit, predict, score
    """
    if clf is None:
      clf = svm.LinearSVC(**SVM_KWARGS)
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
    if isinstance(df_feature, pd.DataFrame):
      feature_vals = df_feature.to_numpy()
      label_vals = ser_label.values
    else:
      feature_vals = df_feature
      label_vals = ser_label
    self.clf.fit(feature_vals, label_vals)
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
    if len(dfs_feature) > 0:
      return dfs_feature[0], ser_label
    else:
      return None, ser_label


##########################################
class MetaClassifierAverage(MetaClassifier):
  # Trains on the average of feature values.

  def _makeTrainingData(self, dfs_feature, ser_label):
    """
    Does fit on the average of the feature values.
    """
    df_feature = dfs_feature[0].copy()
    for df in dfs_feature[1:]:
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
    super().__init__(clf=PluralityClassifier())

  def _makeTrainingData(self, _, ser_label):
    return None, ser_label


##########################################
class MetaClassifierEnsemble(MetaClassifier):
  # Create an ensemble of classifier from each feature replica

  def __init__(self, clf=None,
      is_plurality=True, max_score_std=0.05):
    """
    :param bool is_plurality: Choose most common lablel
                              If false, select label from the
                                distribution of classifier results
    :parm float max_score_std: maximum standard deviation for accuracy score.
    """
    if clf is None:
      clf = svm.LinearSVC(**SVM_KWARGS)
    super().__init__(clf=clf)
    self._is_plurality = is_plurality
    self._max_score_std = max_score_std
    self.ensemble = None  # list-clf
    self._is_deterministic = None  # Label is predicted deterministically

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
    def selectLabelPlurality(ser):
      """
      Selects the most frequently occurring label and randomly
      select among ties.
      :return object, bool: label, deterministic label selection
      """
      # TODO: Efficiency
      ser_count = ser.value_counts()
      ser_top = ser_count[ser_count==ser_count[ser_count.index[0]]]
      idx = np.random.randint(0, len(ser_top))
    
      return ser_top.index[idx], (len(ser_top) == 1)
       
    #
    def selectLabelDistribution(ser):
       """
       Selects the most frequently occurring label and randomly
       select among ties.
       :return object, bool: label, not deterministic
       """
       idx = np.random.randint(0, len(ser))
       return ser.index[idx], False
    #
    self._check()
    self._is_deterministic = False
    # DataFrame where each row is a classifier prediction
    df_label = pd.concat(
        [pd.Series(clf.predict(df_feature)) for clf in self.ensemble],
        axis=1)
    df_label = df_label.T
    # Predictions
    if self._is_plurality:
      func = selectLabelPlurality
    else:
      func = selectLabelDistribution
    results = [func(df_label[c]) for c in df_label.columns]
    predicts = [r[0] for r in results]
    self._is_deterministic = all([r[1] for r in results])
    #
    return pd.Series(predicts)

  def _calcMatch(self, df_feature, ser_label):
    """
    Calculates the number of matches of the instances in score.
    This is done may be done repeatedly.
    :param pd.DataFrame df_feature:
    :param pd.Series ser_label:
    :param int num_iter: number of iterations in the matching
    :return float:
    """
    # TODO: Efficiency
    matches = []
    num_iter = min(MAX_ITER, int((0.5/self._max_score_std)**2))
    for _ in range(num_iter):
      ser_predicted = self.predict(df_feature)
      num_match = sum([1 for c1, c2 in 
          zip(ser_label.values, ser_predicted.values) if c1 == c2])
      matches.append(num_match)
      if self._is_deterministic:
        break
    return np.mean(matches)
