"""Multiple densities with the same variates."""

import pandas as pd
import numpy as np

import common_python.constants as cn
from common_python.plots import util_plots
from common_python.statistics import density


class MultipleDensity(object):
  
  def __init__(self, df, variates):
    """
    Determines the variates present in the cells of the DataFrame.
    :param pd.DataFrame df: data matrix
    :param list-object variates: variate variates
    :param return: a density dataframe; columns are features;
        row index are variates; variates are probabilities ([0,1])
    """
    sers = []
    for col in df.columns:
      density = density.Density(df[col], variates=variates)
      sers.append(density.ser_density)
    self.df = pd.concat(sers, axis=1)

  def calcSortIndex(self, sort_order=None):
    """
    Creates a sort index for each feature based on the combination
    of probability values for each variate.
    Since values are assumed to be in the range [0, 1], the sort index
    consists of successive groups of 3 numerials (int(100*value)).
    :param list-object sort_order:
    :return pd.Series: index is feature, value is sort_index
    """
    if sort_order is None:
      sort_order = self.df.index.tolist()
    else:
      sort_order = list(sort_order)
      sort_order.sort()
    # Calculate the sort value
    sort_values = {}
    for col in df.columns:
      sort_value = 0
      for ordinate in sort_order:
        ordinate_index = int(100*self.df.loc[ordinate, col])
        sort_value += INDEX_MULT*sort_value + ordinate_index
      sort_values[col] = sort_value
    #
    return pd.Series(sort_values)

  def plotMarginals(self, ser_sort_order=None, **plot_opts):
    """
    Does a heatmap of the marginals. X-axis is variates; y-axis are features.
        Values are probabilities.
    :param pd.Series ser_sort_order: Series with features as 
        index with floats defining order. 
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
    setDefault(opts, cn.PLT_XLABEL, "Variate")
    setDefault(opts, cn.PLT_YLABEL, "Feature")
    util_plots.plotCategoricalHeatmap(df.T, **plot_opts)
    return

  def plotMarginalComparisons(self, other, ser_sort_order=None):
    """
    Multiple line plots (one for each variate)
      x-axs probability of a in this distribution
      y-axis probability in other distribution
      point is order pairs of probabilities for the same feature and variate
    :param MultipleDensity other:
    :param list-object sort_order:
    """
    return
