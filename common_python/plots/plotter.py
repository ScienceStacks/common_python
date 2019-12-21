"""Handles common plotting options using axes."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common_python.constants as cn


class Plotter(object):

  def __init__(self, figure=None):
    """
    Initialize the figure
    """
    if figure is None:
      self.figure = plt.figure()
    else:
      self.figure = figure
    self.ax = self.figure.add_subplot(1, 1, 1)
  
  def do(self, is_plot=True, **kwargs):
    """
    Sets options for the plot.
    """
    if  cn.PLT_TITLE in kwargs.keys():
      self.ax.set_title(kwargs[cn.PLT_TITLE])
    if  cn.PLT_XLABEL in kwargs.keys():
      self.ax.set_xlabel(kwargs[cn.PLT_XLABEL])
    if  cn.PLT_YLABEL in kwargs.keys():
      self.ax.set_ylabel(kwargs[cn.PLT_XLABEL])
    if is_plot:
      plt.show()
