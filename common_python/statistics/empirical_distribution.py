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
      df = self._df[col].copy()
      df = df.sort_values()
      df.index = range(len(df))
      dfs.append(df)
    df_result = pd.concat(dfs, axis=1)
    df_result.index = [str(int((100*i)/len(self._df)))
        for i in df_result.index]
    return df_result

  def getProb(self, col, value):
    """
    Computes the probability in the empirical distribution for a
    column.
    :param str col:
    :param float value:
    :return float:
    """
    count = len(self._df[self._df[col] <= value])
    return (1.0*count)/ len(self._df)

  def plot(self, plot_opts=None):
    """
    Creates a heatmap of the marginal distributions.
    x-axis is percentile; y-axis is feature
    :param dict plot_opts:
    """
    def setDefault(opts, key, value):
      if not key in opts.keys():
        opts[key] = value
    #
    if plot_opts is None:
      plot_opts = {}
    df = self.getMarginals()
    opts = dict(plot_opts)
    setDefault(opts, cn.PLT_XLABEL, "Percentile")
    setDefault(opts, cn.PLT_YLABEL, "Feature")
    util_plots.plotCategoricalHeatmap(df.T, **opts)

  def synthesize(self, nobs, frac, **kwargs):
    """
    Returns a random sample of rows with frac values replaced
    by values from the empirical CDF.
    :param int nobs: number of observations in the sample
    :param float frac: fraction of values replaced
    :param dict kwargs: arguments passed to sampler
    :return pd.DataFrame:
    """
