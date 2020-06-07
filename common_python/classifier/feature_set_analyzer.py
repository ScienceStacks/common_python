'''Analyzes alternative collections of classifier features.'''

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
    fset =  feature_analyzer.unmakeFsetStr(fset_stg)
    if len(fset.intersection(selecteds)) > 0:
      continue
    else:
      # Include this feature set
      selecteds.extend(list(fset))
      fset_stgs.append(fset_stg)
  ser_result = ser[fset_stgs].copy()
  return ser_result

def _uk(fset_stg):
  # Converts an fset string into an fset
  return feature_analyzer.unmakeFsetStr(fset_stg)

def _mk(fset):
  # Makes an fset string
  return feature_analyzer.makeFsetStr(fset)


############### CLASSES ####################
class FeatureSetAnalyzer(object):

  def __init__(self, analyzer):
    """
    Parameters
    ----------
    analyzer: FeatureAnalyzer
    """
    self._analyzer = analyzer
    # self.ser_fset
    #  index: string representation of a set of features
    #  value: accuracy (score)
    self.ser_fset = self._analyzer.ser_fset.copy()
    self._reindex()  # Ensure that fsets strings are in order
    self.disjointify()


  def _reindex(self):
    index = self.ser_fset.index.tolist()
    new_index = [_mk(_uk(s)) for s in index]
    self.ser_fset.index = new_index

  def disjointify(self, **kwargs):
    """
    Makes disjoint the feature sets disjoint by discarding
    feature sets that have non-null intersection with a
    more accurate feature set.

    Parameters
    ----------
    kwargs: dict

    Returns
    -------
    None
    """
    self.ser_fset = disjointify(self.ser_fset, **kwargs)

  def _makeCandidateSer(self, fset, min_score=0):
    """
    Creates a Series with the features most likely to
    increase the accuracy of the feature set. The features
    are chosen based on their incremental predication accuracy
    (IPA).

    Parameters
    ----------
    fset : set
      Set of features.

    Returns
    -------
    pd.Series
        index: feature, sorted by descending value
        value: max ipa
    """
    score_dct = {}
    for feature in fset:
      ser_ipa = self._analyzer.df_ipa[feature]
      for other_feature, other_score in ser_ipa.to_dict().items():
        if not other_feature in score_dct:
          score_dct[other_feature] = []
        score_dct[other_feature].append(other_score)
    ser_dct = {k: max(v) for k, v in score_dct.items()}
    ser = pd.Series(ser_dct)
    ser = ser[ser >= min_score]
    ser = ser.drop(list(fset))
    ser = ser.sort_values(ascending=False)
    return ser

  def optimize(self, min_score=MIN_SCORE):
    """
    Optimizes the feature sets to increase their accuracy.
    Each of the final sets should have a score that is no
    less than min_score.

    Parameters
    ----------
    max_score : float
        minimum classification score

    Returns
    -------
    pd.Series
    """
    ser = self._analyzer.ser_sfa
    ser = ser[ser >= min_score]
    process_dct = ser.to_dict()
    result_dct = {}
    #
    def getScore(fset):
      # Gets the score for an fset
      return process_dct[_mk(fset)]
    # Iteratively consider combinations of fsets
    while len(process_dct) > 0:
      cur_fset_stg = list(process_dct.keys())[0]
      cur_fset = _uk(cur_fset_stg)
      cur_score = process_dct[cur_fset_stg]
      if len(process_dct) == 1:
        if cur_score >= min_score:
          result_dct[cur_fset_stg] = getScore(cur_fset)
        del process_dct[cur_fset_stg]
        break
      #
      del process_dct[cur_fset_stg]
      # Look for a high accuracy feature set
      is_changed = False
      for other_fset_stg in process_dct.keys():
        other_fset = _uk(other_fset_stg)
        new_fset = cur_fset.union(other_fset)
        new_score = self._analyzer.score(new_fset)
        old_score =  max(cur_score, getScore(other_fset))
        if new_score < old_score*MIN_FRAC_INCR:
          continue
        if new_score < min_score:
          continue
        # The new feature set improves the classifier
        # Add the new feature; delete the old ones
        new_fset_stg = _mk(new_fset)
        process_dct[new_fset_stg] = new_score
        del process_dct[other_fset_stg]
        is_changed = True
        break
      if not is_changed:
        result_dct[cur_fset_stg] = cur_score
    ser = pd.Series(result_dct)
    ser = ser.sort_values(ascending=False)
    return ser
