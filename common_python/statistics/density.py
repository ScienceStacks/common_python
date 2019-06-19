"""Creation, Analysis, and Manipulation of Discrete Distributions."""

"""
TODO
1. Bar plot of density
"""


import common_python.constants as cn
from common_python.plots import util_plots

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


INDEX_MULT = 100


class Density(object):
  """
  self.ser_density is a Series whose index is the variate
  and the value is its density.
  """

  def __init__(self, ser, variates=None):
    """
    :param pd.Series: variate values for which a density is created
    :param list-object: expected values in the density
    """
    if variates is None:
      variates = ser.unique().tolist()
    self.variates = variates
    self.ser_density = self.__class__._makeDensity(ser, self.variates)

  @staticmethod
  def _makeDensity(ser, variates):
    col = "dummy"
    df = pd.DataFrame({col: ser})
    df_groupby = df.groupby(col)
    groups = df_groupby.groups
    keys = [k for k in groups.keys()]
    keys.extend(variates)
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

  # TODO: Write tests
  def isLessEqual(self, other):
    """
    Determines if lower values have higher probabilities.
    :param Density other:
    :return bool:
    """
    is_less = True
    for key in self.ser_density.keys():
      if is_less:
        if self.ser_density.loc[key][0] >  \
            other.ser_density.loc[key][0]:
          is_less = False
      else:
        if self.ser_density.loc[key][0] <  \
            other.ser_density.loc[key][0]:
          return False
    return True

  def plot(self, is_plot=True, **kwds):
    """
    Creates a bar plot of the density.
    """
    self.ser_density.plot.bar(**kwds)
    if is_plot:
      plt.show()
