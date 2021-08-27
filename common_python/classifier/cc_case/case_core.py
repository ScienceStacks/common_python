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

  def __init__(self, feature_vector, fv_statistic, dtree=None):
    """
    Parameters
    ----------
    feature_vector: FeatureVector
    fv_statistic: FeatureVectorStatistic
    dtree: sklearn.DecisionTreeClassifier
        Tree from which case was constructed
    """
    self.feature_vector = feature_vector
    self.fv_statistic = fv_statistic
    self.dtree = dtree

  def __repr__(self):
    return "%s-- %s" % (str(self.feature_vector), str(self.fv_statistic))

  def __eq__(self, other_case):
    if str(self) != str(other_case):
      return False
    return self.fv_statistic == other_case.fv_statistic


##################################################################
class CaseCollection(dict):
  """
  key: FeatureVector in sorted order
  value: Case
  """

  def sort(self):
    """
    Sorts the dictionary by key.
    """
    keys = list(self.keys())
    # Check if a sort is needed
    is_sorted = True
    for key1, key2 in zip(keys[0:-1], keys[1:]):
      if key1 > key2:
        is_sorted = False
        break
    if is_sorted:
      return
    #
    keys.sort()
    dct = {k: self[k] for k in keys}
    [self.__delitem__(k) for k in keys]
    self.update(dct)

  def __eq__(self, other_col):
    diff = set(self.keys()).symmetric_difference(other_col.keys())
    if len(diff) > 0:
      return False
    trues = [self[k] == other_col[k] for k in self.keys()]
    return all(trues)

  def toDataframe(self):
    """
    Creates a dataframe from the data in the cases.

    Returns
    -------
    pd.DataFrame
        index: str(feature_vector)
        columns: cn.NUM_POS, cn.NUM_POS, cn.SIGLVL
    """
    siglvls = [c.fv_statistic.siglvl for c in self.values()]
    num_samples = [c.fv_statistic.num_sample for c in self.values()]
    num_poss = [c.fv_statistic.num_pos for c in self.values()]
    prior_probs = [c.fv_statistic.prior_prob for c in self.values()]
    df = pd.DataFrame({
        cn.SIGLVL: siglvls,
        cn.PRIOR_PROB: prior_probs,
        cn.NUM_SAMPLE: num_samples,
        cn.NUM_POS: num_poss,
        }, index=list(self.keys()))
    return df.sort_index()

  def _checkCommon(self, other_col):
    if IS_CHECK:
      common_stg = list(set(self.keys()).intersection(other_col.keys()))
      trues = [self[k] == other_col[k] for k in common_stg]
      if not all(trues):
        raise RuntimeError("Common Cases are not equal.")

  def union(self, other_col):
    """
    Union of two CaseCollection.

    Parameters
    ----------
    other_col: CaseCollection

    Returns
    -------
    CaseCollection
    """
    self._checkCommon(other_col)
    case_col = copy.deepcopy(self)
    case_col.update(other_col)
    case_col.sort()
    return case_col

  def intersection(self, other_col):
    """
    Intersection of two CaseCollection.

    Parameters
    ----------
    other_col: CaseCollection

    Returns
    -------
    CaseCollection
    """
    self._checkCommon(other_col)
    common_keys = set(self.keys()).intersection(other_col.keys())
    cases = [self[k] for k in common_keys]
    return CaseCollection.make(cases)

  def difference(self, other_col):
    """
    Difference of two CaseCollection.

    Parameters
    ----------
    other_col: CaseCollection

    Returns
    -------
    CaseCollection
    """
    self._checkCommon(other_col)
    difference_keys = set(self.keys()).difference(other_col.keys())
    cases = [self[k] for k in difference_keys]
    return CaseCollection.make(cases)

  # TESTME
  def symmetricDifference(self, other_col):
    """
    What's not common to both.

    Parameters
    ----------
    other_col: CaseCollection

    Returns
    -------
    CaseCollection
    """
    self._checkCommon(other_col)
    case_col = self.difference(other_col)
    case_col.extend(other_col.differeince(self))
    return CaseCollection.make(cases)

  ################### CLASS METHODS ###################
  @classmethod
  def make(cls, cases):
    """
    Returns sorted CaseCollection.

    Parameters
    ----------
    list-Case

    Returns
    -------
    CaseCollection (sorted)
    """
    case_col = CaseCollection({str(c.feature_vector): c for c in cases})
    case_col.sort()
    return case_col
