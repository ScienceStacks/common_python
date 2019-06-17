"""Manipulation of Empirical Distributions."""

"""
DataFrames are structured so that columns are features and
rows are instances (observations).
"""

import pandas as pd
import numpy as np

import common_python.constants as cn
from common_python.plots import util_plots


class EmpiricalDistribution(object):

  def __init__(self, df):
    """
    :param pd.DataFrame: empirical distribution
    """
    self._df = df

  def sample(self, nobs, is_decorrelate=True):
    """
    Samples with replacement.
    :param int nobs:
    """
    df_sample = self._df.sample(nobs, replace=True)
    if is_decorrelate:
      df_sample = self.__class__.decorrelate(df_sample)
    df_sample = df_sample.reset_index()
    return df_sample

  @staticmethod
  def decorrelate(df):
    """
    Permutes rows within columns to remove correlations between features.
    :return pd.DataFrame:
    """
    length = len(df)
    df_result = df.copy()
    for col in df_result.columns:
      values = df_result[col].tolist()
      df_result[col] = np.random.permutation(values)
    return df_result
      
    

  def getMarginals(self):
    """
    Constructs the marginal distributions for all columns.
    :return pd.DataFrame: index is fraction
    """
    dfs = []
    for col in self._df.columns:
      dfs.append(self._df[col].sort_values())
    df_result = pd.concat(dfs, axis=1)
    df_result.index = [str((100*i)/len(dfs)) for i in df_result.index]

  def getProb(self, cols, values):
    """
    Computes the probability in the empirical distribution for columns.
    :param list-str cols:
    :param list-float values:
    :return float:
    """
    df_sub = self._df[cols]
    count = 0
    indicies = self._df.index
    for idx in indices:
      for pos, col in enumerate(self._df.columns):
        ele = self._df.loc[idx, col]
        if values[pos] > ele:
          next
      count += 1
    return (1.0*count)/len(self._df)

  def plot(self):
    """
    Creates a heatmap of the distributions.
    x-axis is percentile; y-axis is feature
    """
    df = self.getMarginals()
    opts =  {
        cn.PLT_XAXIS: "Percentile",
        cn.PLT_YAXIS: "Feature",
        }
    util_plots.plotCategoricalHeatmap(df.T, **opts)
