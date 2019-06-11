"""Plot Utilities."""

import common_python.constants as cn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Plot options
XLABEL = "xlabel"
YLABEL = "ylabel"
TITLE = "title"

def getAxis(ax):
  if ax is None:
    ax = plt.gca()
  return ax

def plotTrinaryHeatmap(df, ax=None, is_plot=True, **kwargs):
  """
  Plots a heatmap for a dataframe with trinary values: -1, 1, 0
  :param plt.Axis ax:
  :param bool is_plot: shows plot if True
  :param dict kwargs: plot options
  """
  # Setup the data
  df_plot = df.applymap(lambda v: np.nan if np.isclose(v, 0)
      else v)
  # Plot construct
  if ax is None:
    plt.figure(figsize=(16, 10))
  ax = getAxis(ax)
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

def plotCategoricalHeatmap(df, is_plot=True, **kwargs):
  """
  Plots a heatmap of numerical values with categorical
  x and y axes.
  Row indices are the y-axis; columns are the x-axis
  :param pd.DataFrame df:
  :param dict kwargs: plot options
  """
  def setValue(key, func):
    if key in kwargs.keys():
      func(kwargs[key])
  #
  if cn.PLT_FIGSIZE in kwargs.keys():
    plt.figure(figsize=kwargs[cn.PLT_FIGSIZE])
  ax = plt.gca()
  ax.set_xticks(np.arange(len(df.columns))+0.5)
  ax.set_xticklabels(df.columns)
  ax.set_yticks(np.arange(len(df.index))+0.5)
  ax.set_yticklabels(df.index)
  heatmap = plt.pcolor(df, cmap='jet')
  plt.colorbar(heatmap)
  setValue(cn.PLT_XLABLE, plt.xlabel)
  setValue(cn.PLT_YLABLE, plt.ylabel)
  setValue(cn.PLT_TITLE, plt.title)
  if cn.PLT_IS_PLOT in kwargs.keys():
    if kwargs[cn.PLT_IS_PLOT]:
      plt.show()
  else:
    plt.show()
