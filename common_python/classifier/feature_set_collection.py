'''Constructs and evaluates alternative selections of classifier features.'''

import common_python.constants as cn
from common_python.classifier import util_classifier
from common_python.classifier import feature_analyzer

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn

MIN_FRAC_INCR = 1.01  # Must increase by at least 1%
MIN_SCORE = 0
FEATURE_SEPARATOR = "+"

############ FUNCTIONS #################
def disjointify(ser_fset, min_score=MIN_SCORE):
  """
  Makes disjoint the feature sets disjoint by discarding
  feature sets that have non-null intersection with a
  more accurate feature set.

  Parameters
  ----------
  ser_fset: pd.Series
  max_score : float
      minimum classification score

  Returns
  -------
  pd.Series
  """
  ser = ser_fset[ser_fset >= min_score]
  ser = ser.copy()
  selecteds = []  # Features selected
  fset_stgs = []  # Feature strings selected
  for fset_stg in ser.index:
    fset =  FeatureSet(fset_stg)
    if len(fset.set.intersection(selecteds)) > 0:
      continue
    else:
      # Include this feature set
      selecteds.extend(list(fset.set))
      fset_stgs.append(fset.str)
  ser_result = ser[fset_stgs].copy()
  return ser_result


############### CLASSES ####################
class FeatureSet(object):
  '''Represetation of a set of features'''

  def __init__(self, descriptor):
    """
    Parameters
    ----------
    descriptor: list/set/string
    """
    if isinstance(descriptor, str):
      self.str = descriptor
      self.set = FeatureSet._unmakeStr(self.str)
    elif isinstance(descriptor, set)  \
        or isinstance(descriptor, list):
      self.set = set(descriptor)
      self.str = FeatureSet._makeStr(self.set)
    else:
      raise ValueError("Invalid argument type")

  def __str__(self):
    return self.str

  @staticmethod
  def _makeStr(features):
    """
    Creates a string for a feature set
    :param iterable-str features:
    :return str:
    """
    f_list = list(features)
    f_list.sort()
    return cn.FEATURE_SEPARATOR.join(f_list)
  
  @staticmethod
  def _unmakeStr(fset_stg):
    """
    Recovers a feature set from a string
    """
    return set(fset_stg.split(cn.FEATURE_SEPARATOR))

  def equals(self, other):
    diff = self.set.symmetric_difference(other.set)
    return len(diff) == 0


########################################  
class FeatureSetCollection(object):

  def __init__(self, analyzer):
    """
    Parameters
    ----------
    analyzer: FeatureAnalyzer
    """
    self._analyzer = analyzer
    self._ser_fset = None  # Use self.make() to construct
    #  index: string representation of a set of features
    #  value: accuracy (score)

  @property
  def ser_fset(self):
    """
    Calculates the classification accuracy of feature sets
    of size 1 and 2.
    :return pd.Series: Sorted by descending accuracy
        Index: feature set (feature separated by "+")
        Value: Accuracy
    """
    if self._ser_fset is None:
      # Feature sets of size 1
      ser1 = self._analyzer.ser_sfa.copy()
      # Feature sets of size two
      feature_sets = []
      accuracies = []
      for idx, feature1 in enumerate(
          self._analyzer.features):
        for feature2 in self._analyzer.features[(idx+1):]:
          fset = FeatureSet([feature1, feature2])
          feature_sets.append(fset.str)
          try:
            accuracy = self._analyzer.df_ipa.loc[
                feature1, feature2] +  \
                max(self._analyzer.ser_sfa.loc[feature1],
                    self._analyzer.ser_sfa.loc[feature2])
          except KeyError:
            accuracy = np.nan  # Handle missing keys
          accuracies.append(accuracy)
      ser2 = pd.Series(accuracies, index=feature_sets)
      # Construct result
      self._ser_fset = pd.concat([ser1, ser2])
      self._ser_fset = self._ser_fset.sort_values(
          ascending=False)
    return self._ser_fset

  def disjointify(self, **kwargs):
    """
    Creates a list of feature set strings with non-overlapping
    features.

    Parameters
    ----------
    kwargs: dict
        Parameters passed to function.

    Returns
    -------
    pd.Series
    """
    return disjointify(self.ser_fset, **kwargs)
    

  def _makeCandidateSer(self, fset, min_score=0):
    """
    Creates a Series with the features most likely to
    increase the accuracy of the feature set. The features
    are chosen based on their incremental predication accuracy
    (IPA).

    Parameters
    ----------
    fset : FeatureSet
      Set of features.

    Returns
    -------
    pd.Series
        index: feature, sorted by descending value
        value: max ipa
    """
    score_dct = {}
    for feature in fset.set:
      ser_ipa = self._analyzer.df_ipa[feature]
      for other_feature, other_score in ser_ipa.to_dict().items():
        if not other_feature in score_dct:
          score_dct[other_feature] = []
        score_dct[other_feature].append(other_score)
    ser_dct = {k: max(v) for k, v in score_dct.items()}
    ser = pd.Series(ser_dct)
    ser = ser[ser >= min_score]
    ser = ser.drop(list(fset.set))
    ser = ser.sort_values(ascending=False)
    return ser

  def optimize(self, min_score=MIN_SCORE):
    """
    Optimizes the collection of features sets by
    finding increases in score accuracy.

    Parameters
    ----------
    max_score : float
        minimum classification accuracy score

    Returns
    -------
    pd.Series
    """
    ser = self.disjointify(min_score=min_score)
    process_dct = ser.to_dict()
    result_dct = {}
    #
    def getScore(fset):
      # Gets the score for an fset
      return process_dct[fset.str]
    # Iteratively consider combinations of fsets
    while len(process_dct) > 0:
      cur_fset = FeatureSet(list(process_dct.keys())[0])
      cur_score = process_dct[cur_fset.str]
      if len(process_dct) == 1:
        if cur_score >= min_score:
          result_dct[cur_fset.str] = getScore(cur_fset)
        del process_dct[cur_fset.str]
        break
      #
      del process_dct[cur_fset.str]
      # Look for a high accuracy feature set
      is_changed = False
      for other_fset_stg in process_dct.keys():
        other_fset = FeatureSet(other_fset_stg)
        new_fset = FeatureSet(
            cur_fset.set.union(other_fset.set))
        new_score = self._analyzer.score(new_fset.set)
        old_score =  max(cur_score, getScore(other_fset))
        if new_score < old_score*MIN_FRAC_INCR:
          continue
        if new_score < min_score:
          continue
        # The new feature set improves the classifier
        # Add the new feature; delete the old ones
        process_dct[new_fset.str] = new_score
        del process_dct[other_fset.str]
        is_changed = True
        break
      if not is_changed:
        result_dct[cur_fset.str] = cur_score
    ser = pd.Series(result_dct)
    ser = ser.sort_values(ascending=False)
    return ser
