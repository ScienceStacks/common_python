"""Handles common plotting options using axes."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common_python.constants as cn

DEF_SUBPLOTS = [(1, 1, 1)]

  

class Plotter(object):

  def __init__(self, figure=None, subplots=DEF_SUBPLOTS):
    """
    Initialize the figure
    """
    if figure is None:
      self.figure = plt.figure()
    else:
      self.figure = figure
    self.axes = []
    for subplot in subplots:
      self.axes.append(self.figure.add_subplot(*subplot))
    self.ax = self.axes[0]
    self.kwargs = {}

  def doAllAx(self, **kwargs):
    for ax in self.axes:
      self.doAx(ax, **kwargs)

  def mergeOptions(self, kwargs):
    """
    Merges with default options.
    :param dict kwargs:
    :return dict:
    """
    new_kwargs = dict(kwargs)
    for key in self.kwargs:
      if not key in kwargs:
        new_kwargs[key] = self.kwargs[key]
    return new_kwargs

  def doAx(self, ax, **kwargs):
    new_kwargs = self.mergeOptions(kwargs)
    if  cn.PLT_TITLE in new_kwargs.keys():
      ax.set_title(new_kwargs[cn.PLT_TITLE])
    if  cn.PLT_XLABEL in new_kwargs.keys():
      ax.set_xlabel(new_kwargs[cn.PLT_XLABEL])
    if  cn.PLT_YLABEL in new_kwargs.keys():
      ax.set_ylabel(new_kwargs[cn.PLT_YLABEL])
    if  cn.PLT_XLIM in new_kwargs.keys():
      ax.set_xlim(new_kwargs[cn.PLT_XLIM])
    if  cn.PLT_YLIM in new_kwargs.keys():
      ax.set_ylim(new_kwargs[cn.PLT_YLIM])
    if  cn.PLT_XTICKLABELS in new_kwargs.keys():
      ax.set_xticklabels(new_kwargs[cn.PLT_XTICKLABELS])
    if  cn.PLT_YTICKLABELS in new_kwargs.keys():
      ax.set_yticklabels(new_kwargs[cn.PLT_YTICKLABELS])
    if  cn.PLT_LEGEND in new_kwargs.keys():
      leg = ax.legend(new_kwargs[cn.PLT_LEGEND])

  def setDefault(self, keyword, default):
    """
    Sets the option to a default value if the
    keyword is not present.
    :param dict kwargs:
    :param str keyword:
    :param object default:
    """
    self.kwargs[keyword] = default
  
  def do(self, is_plot=True, **kwargs):
    """
    Sets options for the plot.
    """
    for ax in self.axes:
      self.doAx(ax, **kwargs)
    if is_plot:
      plt.show()
