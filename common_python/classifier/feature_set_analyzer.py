'''Analyzes alternative collections of classifier features.'''

import common_python.constants as cn
from common_python.classifier import util_classifier
from common_python.classifier import feature_analyzer
from common_python.util import util
from common_python.util.persister import Persister

import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn


class FeatureSetAnalyzer(object):

  def __init__(self, ser_fset):
    """
    Parameters
    ----------
    ser_fset : pd.Series
      Index (str): feature set indicated by a "+" separator.
      Value (float): classification accuracy.
      Sort: descending value
    """
    self._ser_fset = ser_fset

  def _makeNonIntersectingSets(self, min_accuracy):
    """
    Prunes the feature sets so that they are non-intersecting.

    Parameters
    ----------
    min_accuracy: float
      Minimum accuracy for a set to be included.

    Returns
    -------
    list-str.
      List of string representation of sets
    """
    ser = self._ser_fset[self._ser_fset >= min_accuracy]
    features = []  # Features selected
    indices = []  # Feature strings selected
    for idx in ser.index:
      this_features = feature_str.split(
          feature_analyzer.FEATURE_SEPARATOR)
      if len(set(this_features).intersection(features)) > 0:
        continue
      else:
        features.extend(features)
        indices.append(idx)
    return indices



