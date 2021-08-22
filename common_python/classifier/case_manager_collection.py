'''Manages Cases and case queries for multiple classes'''

from common_python.classifier.case_manager import CaseManager

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


############################################
class CaseManagerCollection:

  # Manages data with non-binary classes

  def __init__(self, df_X, ser_y, **kwargs):
    """
    Parameters
    ----------
    df_X: pd.DataFrame
        columns: features
        index: instance
        value: feature value
    ser_y: pd.Series
        index: instance
        value: non-binary classes
    kwargs: dict
        optional arguments used for CaseManager
    """
    self.manager_dct = {}
    self.classes = list(set(ser_y.values))
    for a_class in self.classes:
      new_ser_y = ser_y.apply(lambda v: 1 if v == a_class else 0)
      self.manager_dct[a_class] = CaseManager(df_X, new_ser_y, **kwargs)

  def build(self):
    for a_class in self.classes:
      self.manager_dct[a_class].build()

  # FIXME: Appears that feature vectors are duplicated from the heatmap
  #        but it's not in the  datafra,e
  def plotHeatmap(self, ax=None, is_plot=True):
    """
    Constructs a heatmap in which x-axis is state, y-axis is feature,
    value is significance level using a temperature color code.

    Returns
    -------
    pd.DataFrame
        columns: class
        index: feature
        value: significance level
    """
    if ax is None:
      _, ax = plt.subplots(1)
    max_sl = 5
    # Contruct a datadrame
    sers = [m.toSeries() for m in self.manager_dct.values()]
    df = pd.concat(sers, axis=1)
    df.columns = list(self.manager_dct.keys())
    # Convert to number of zeros
    df = df.applymap(lambda v: CaseManager.convertSLToNumzero(v))
    df_plot = df.copy()
    df_plot.index = list(range(len(df_plot)))
    # Do the plot
    sns.heatmap(df_plot, cmap='seismic', ax=ax, vmin=-max_sl, vmax=max_sl)
    ax.set_ylabel = "feature vector"
    ax.set_xlabel = "class"
    #
    if is_plot:
      plt.show()
    return df
