'''
Assign values to FeatureSets features and perform
operations on these features (e.g., XOR).

Feature values are integers in [-1, 0, 1].
Distances between features are integers in [0, 1, 2]
'''

import common_python.constants as cn
from common_python.classifier import util_classifier
from common_python.util import util
from common_python.classifier  \
    import feature_set_collection

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


class FeatureSetVector(object):

  def __init__(self, analyzer, feature_sets, ser_X):
    """
    Parameters
    ----------
    analyzer: FeatureAnalyzer
    feature_sets: list-FeatureSet
    ser_X: pd.Series
        index: feature (str)
        value: score (float)
  
    Returns
    -------
    None.
    """
    ##### PRIVATE #####
    self._analyzer = analyzer
    self._sets = list(feature_sets)
    self._ser_X = ser_X
    self._df_value = None
    self._features = None

  @property
  def features(self):
    if self._features is None:
      self._features = self.df_value[
          cn.FEATURE].tolist()
    return self._features

  @property
  def df_value(self):
    """
    Creates a dataframe that represents the
    feature set vector
  
    Returns
    -------
    pd.DataFrame
        columns: 
          cn.FEATURE_SET (str)
          cn.FEATURE (str)
          cn.VALUE (float)
    """
    if self._df_value is None:
      self._df_value = self._make_df_value()
    return self._df_value

  def _make_df_value(self):
    """
    Creates a dataframe that represents the
    feature set vector
  
    Returns
    -------
    pd.DataFrame
        columns: 
          cn.FEATURE_SET (str)
          cn.FEATURE (str)
          cn.VALUE (float)
    """
    dct = {cn.FEATURE_SET: [], cn.FEATURE: [], 
        cn.VALUE: []}
    for fset_stg in self._sets:
      fset = feature_set_collection.FeatureSet(fset_stg)
      for feature in fset.set:
        dct[cn.FEATURE_SET].append(fset_stg)
        dct[cn.FEATURE].append(feature)
        dct[cn.VALUE].append(self._ser_X.loc[feature])
    df = pd.DataFrame(dct)
    return df

  def xor(self, other):
    """
    Calculates a distance between 
  
    Parameters
    ----------
    other: FeatureSetVector
  
    Returns
    -------
    FeatureSetVector.
    """
    if not util.isSetEqual(self.features, other.features):
      raise ValueError("Must have the same features.")
    df_value = other.df_value.copy()
    df_value[cn.VALUE] = -1* df_value[cn.VALUE]
    df = pd.concat([self.df_value, df_value])
    df = df[[cn.FEATURE, cn.VALUE]]
    ser = df.groupby(cn.FEATURE).sum()
    return FeatureSetVector(self._analyzer,
        self.features, ser)

    
