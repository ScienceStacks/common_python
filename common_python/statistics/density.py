"""Creation, Analysis, and Manipulation of Discrete Distributions."""

"""
TODO
1. Bar plot of density
"""


import pandas as pd
import numpy as np

import common_python.constants as cn
from common_python.plots import util_plots


INDEX_MULT = 100


class Density(object):

  def __init__(self, ser, variates=None):
    """
    :param pd.Series: values for which a density is created
    :param list-object: expected values in the density
    """
    self.variates = variates
    self.ser_density = self._makeDensity(ser)

  def _makeDensity(self, ser):
    ser_groupby = ser.groupby()
    groups = ser_groupby.groups
    keys = groups.keys()
    if self.variates is not None:
      keys.extend(self.variates)
      keys = list(set(keys))
    keys.sort()
    density = {}
    for key in keys:
      if not key in groups.keys():
        numr = 0.0
      else:
        numr = 1.0*len(groups[key])
      density[key] = numr/len(ser)
    return pd.Series(density, index=keys)

  def get(self):
    return self.ser_density

  def isLessEqual(self, other):
    """
    Determines if lower values have higher probabilities.
    :param Density other:
    :return bool:
    """
    is_less = True
    for key in self.ser_density.keys():
      if_less:
        if self.ser_density.loc[key][0] >  \
            other.ser_density.loc[key][0]:
          is_less = False
      else:
        if self.ser_density.loc[key][0] <  \
            other.ser_density.loc[key][0]:
          return False
    return True

    

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

  # TODO: Sort y axis?
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
