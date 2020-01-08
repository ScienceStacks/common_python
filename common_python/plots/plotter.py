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

  def doAllAx(self, **kwargs):
    for ax in self.axes:
      self.doAx(ax, **kwargs)

  def doAx(self, ax, **kwargs):
    if  cn.PLT_TITLE in kwargs.keys():
      ax.set_title(kwargs[cn.PLT_TITLE])
    if  cn.PLT_XLABEL in kwargs.keys():
      ax.set_xlabel(kwargs[cn.PLT_XLABEL])
    if  cn.PLT_YLABEL in kwargs.keys():
      ax.set_ylabel(kwargs[cn.PLT_YLABEL])
    if  cn.PLT_XLIM in kwargs.keys():
      ax.set_xlim(kwargs[cn.PLT_XLIM])
    if  cn.PLT_YLIM in kwargs.keys():
      ax.set_ylim(kwargs[cn.PLT_YLIM])
    if  cn.PLT_XTICKLABELS in kwargs.keys():
      ax.set_xticklabels(kwargs[cn.PLT_XTICKLABELS])
    if  cn.PLT_YTICKLABELS in kwargs.keys():
      ax.set_yticklabels(kwargs[cn.PLT_YTICKLABELS])
    if  cn.PLT_LEGEND in kwargs.keys():
      ax.legend(kwargs[cn.PLT_LEGEND])
  
  def do(self, is_plot=True, **kwargs):
    """
    Sets options for the plot.
    """
    for ax in self.axes:
      self.doAx(ax, **kwargs)
    if is_plot:
      plt.show()
