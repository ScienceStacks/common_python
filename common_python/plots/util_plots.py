"""Plot Utilities."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Plot options
XLABEL = "xlabel"
YLABEL = "ylabel"
TITLE = "title"


def plotTrinaryHeatmap(df, is_plot=True, **kwargs):
  """
  Plots a heatmap for a dataframe with trinary values: -1, 1, 0
  :param bool is_plot: shows plot if True
  :param dict kwargs: plot options
  """
  # Setup the data
  df_plot = df.applymap(lambda v: np.nan if np.isclose(v, 0)
      else v)
  # Plot construct
  plt.figure(figsize=(16, 10))
  ax = plt.gca()
  columns = df_plot.columns
  ax.set_xticks(np.arange(len(columns)))
  ax.set_xticklabels(columns)
  heatmap = plt.pcolor(df_plot, cmap='jet')
  plt.colorbar(heatmap)
  if XLABEL in kwargs:
    plt.xlabel(kwargs[XLABEL])
  if YLABEL in kwargs:
    plt.ylabel(kwargs[YLABEL])
  if TITLE in kwargs:
    plt.title(kwargs[TITLE])
  if is_plot:
    plt.show()
