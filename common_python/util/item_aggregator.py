"""Aggregates properties of items in list."""

import common_python.constants as cn

import pandas as pd
import numpy as np

class ItemAggregator(object):

  def __init__(self, func):
    """
    :param UnaryFunction func:
    """
    self.func = func
    self.sers = []
    self._df = None

  def append(self, items):
    self.sers.append(pd.Series([self.func(s) for s in items]))

  @property
  def df(self):
    if self._df is None:
      df_agg = pd.concat(self.sers, axis=1)
      self._df = pd.DataFrame({
          cn.MEAN: df_agg.mean(axis=1),
          cn.STD: df_agg.std(axis=1),
          })
    return self._df
