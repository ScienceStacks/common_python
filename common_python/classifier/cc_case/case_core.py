'''Core classes for case manipulation'''

"""
A Case is a statistically significant FeatureVector in that it distinguishes
between classes.
Herein, only binary classes are considered (0, 1).

A CaseCollection is a collection of Cases that are accessed by
their FeatureValue string. The class exposes ways to interrogate cases
and construct new cases.

The significance level of a case is calculaed based on the frequency of
case occurrence for labels using a binomial null distribution with p=0.5.

#TODO:
1. Remove contradictory cases (positive SL on > 1 class)
2. Handle correlated cases
"""

import common_python.constants as cn
from common_python.classifier.feature_set import FeatureVector

import copy
import numpy as np
import pandas as pd


MAX_SL = 0.05
MIN_SL = 1e-5  # Minimum value of significance level used
               # in conerting to nuumber of zeroes
# Random forest defaults
RANDOM_FOREST_DEFAULT_DCT = {
    "n_estimators": 200,  # Number of trees created
    "max_depth": 4,
    "random_state": 0,
    "bootstrap": False,
    "min_impurity_decrease": 0.01,
    "min_samples_leaf": 5,
    }
TREE_UNDEFINED = -2
IS_CHECK = True  # Does additional tests of consistency


##################################################################
class FeatureVectorStatistic:

  def __init__(self, num_sample, num_pos, prior_prob, siglvl):
    self.siglvl = siglvl
    self.num_sample = num_sample
    self.prior_prob = prior_prob
    self.num_pos = num_pos

  def __repr__(self):
    return "smp=%d, pos=%d, p=%2.2f, sl=%2.2f" % (self.num_sample,
        self.num_pos, self.prior_prob, self.siglvl)

  def __eq__ (self, other_statistic):
    if not np.isclose(self.siglvl, other_statistic.siglvl):
      return False
    if self.num_sample != other_statistic.num_sample:
      return False
    if not np.isclose(self.prior_prob, other_statistic.prior_prob):
      return False
    return self.num_pos == other_statistic.num_pos


##################################################################
class Case:
  """Case for a binary classification."""

  def __init__(self, feature_vector, fv_statistic, dtree=None,
      df_X=None, ser_y=None):
    """
    Parameters
    ----------
    feature_vector: FeatureVector
    fv_statistic: FeatureVectorStatistic
    dtree: sklearn.DecisionTreeClassifier
        Tree from which case was constructed
    df_X: pd.DataFrame - training data
        columns: features
        index: instances
    ser_y: pd.series
        binary class values (0, 1)
    """
    self.feature_vector = feature_vector
    self.fv_statistic = fv_statistic
    self.dtree = dtree
    self.df_X = df_X
    self.ser_y = ser_y
    self.instances = self._getCompatibleInstances(df_X, self.feature_vector)
    if self.instances is None:
      self.instance_str = ""
    else:
      self.instance_str = "+".join(self.instances)

  @staticmethod
  def _getCompatibleInstances(df_X, feature_vector):
    """
    Constructs a bit vector that represents the rows (instances) for which
    the case is compatible.

    Parameters
    ----------
    df_X: pd.DataFrame (features)
    feature_vector: FeatureVector

    Returns
    -------
    list (sorted)
    """
    if df_X is not None:
      instances = []
      for idx in df_X.index:
        ser = df_X.loc[idx,:]
        fv = FeatureVector(ser)
        if fv.contains(feature_vector):
          instances.append(idx)
      instances.sort()
      return instances
    else:
      return None
    

  def __repr__(self):
    return "%s-- %s" % (str(self.feature_vector), str(self.fv_statistic))

  def __eq__(self, other_case):
    if str(self) != str(other_case):
      return False
    return self.fv_statistic == other_case.fv_statistic
