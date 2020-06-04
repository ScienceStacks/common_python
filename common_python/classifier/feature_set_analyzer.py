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
MIN_SCORE = 0.5

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
    fset =  feature_analyzer.unMakeFsetStr(fset_stg)
    if len(fset.intersection(selecteds)) > 0:
      continue
    else:
      # Include this feature set
      selecteds.extend(list(fset))
      fset_stgs.append(fset_stg)
  ser_result = ser[fset_stgs].copy()
  return ser_result


class FeatureSetAnalyzer(object):

  def __init__(self, analyzer):
    """
    Parameters
    ----------
    analyzer: FeatureAnalyzer
    """
    self._analyzer = analyzer
    self.ser_fset = self._analyzer.ser_fset

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

  def combine(self, min_score=MIN_SCORE):
    """
    Extends self.ser_fset by considering binary
    combinations of existing feature sets.
    To be considered, the combined fset
    must have a larger accuracy than the maximum of
    the accuracies of the individual fsets.

    Parameters
    ----------
    max_size : int
        maximum size of a feature set
    max_score : float
        minimum classification score

    Returns
    -------
    pd.Series
    """
    # Initializations for search
    fset_stgs = []
    scores = []
    ser_fset = self.ser_fset[self.ser_fset >= min_score]
    fset_stgs = ser_fset.index.tolist()
    # Consider each in combination
    for fset_stg1 in fset_stgs:
      fset1 = feature_analyzer.unMakeFsetStr(fset_stg1)
      for fset_stg2 in fset_stgs[1:]:
        thr_accuracy = max(ser_fset.loc[fset_stg1],
            ser_fset.loc[fset_stg2])*MIN_FRAC_INCR
        fset2 = feature_analyzer.unMakeFsetStr(fset_stg2)
        new_fset = list(fset1.union(fset2))
        score = self._analyzer.score(new_fset)
        if score >= thr_accuracy:
          scores.append(score)
          fset_stgs.append(
              feature_analyzer.makeFsetStr(new_fset))
    #
    ser = pd.Series(scores, index=fset_stgs)
    ser.sort_values(ascending=False)
    return ser