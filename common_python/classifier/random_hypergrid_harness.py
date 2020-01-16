"""
Experimental harness for evaluating classifiers in a multidimensional
grid (hypergrid). Ground truth is a set of randomly selected
points based on the distribution of values in the coordinate space.
"""

from common_python.classifier.hypergrid_harness import  \
    HypergridHarness, Vector, Plane, TrinaryClassification
import common_python.constants as cn
import common_python.util.util as util
from common_python.util.item_aggregator import ItemAggregator
from common_python.plots.plotter import Plotter

import collections
import copy
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


########### CONSTANTS ################
# Default values
DEF_STDS = [1.0, 1.0]

# Constants
POS = 1
NEG = -1
SMALL = 1e-5

# Parameters
MAX_ITER = 50
THR_IMPURITY = 0.05


class RandomHypergridHarness(HypergridHarness):

  def __init__(self, num_point=25, stds=DEF_STDS, impurity=0.0):
    """
    :param int num_point: number of points in grid
    :param list-float stds: standard deviations for each coordinate
    :param float impurity: in [-1, 1] - difference between pos & neg
    """
    self._stds = stds
    self._num_dim = len(stds)
    self._num_point = num_point
    self._impurity = impurity
    self._plane = Plane(Vector(np.repeat(1, self._num_dim)), offset=0)
    self.trinary = self._makeTrinary()  # Adjusts self._plane
    self._xlim, self._ylim = self._makeAxesLimits() 

  def _makeAxesLimits(self):
    """
    Finds the range of values for the first two axes.
    """
    def makeRange(arr):
      """
      Returns minimum and maximum values of the array.
      """
      return np.round(np.min(arr),2), np.round(np.max(arr), 2)
    #
    arr = np.concatenate([self.trinary.pos_arr, self.trinary.neg_arr])
    xlim = makeRange(arr[:, 0])
    ylim = makeRange(arr[:, 1])
    return xlim, ylim

  def _makeTrinary(self):
    """
    Creates the grid and the 
    labels for the grid based on the classification vector.
    Finds a plane that achieves the desired impurity.
    :return TrinaryClassification Plane:
    """
    OFFSET_ADJ = 1
    OFFSET_ADJ_REDUCE = 0.5  # Amount by which adjustment is reduced
                             # when there is a direction change
    def adjustOffset(cur_impurity, cur_offset, cur_offset_adj):
      """
      Returns new value of the plane adjustment
      :param float cur_impurity:
      :param float cur_offset:
      :param float cur_offset_adj: Amount by which offset is adjusted
      :return float, float: new_offset, new_offset_adj
      """
      # A positive difference means that offset should increase.
      # A negative differences means that offset should decrease.
      diff = cur_impurity - self._impurity
      if diff*cur_offset_adj > 0:
        # Going in the correct direction
        new_offset_adj = cur_offset_adj 
      else:
        # Passed the desired value
        new_offset_adj = -offset_adj*OFFSET_ADJ_REDUCE
      new_offset = cur_offset + cur_offset_adj
      return new_offset, new_offset_adj
    # Try several grids    
    for _ in range(MAX_ITER):
      best_trinary = None
      grid = [np.random.normal(0, std, self._num_point)
          for std in self._stds]
      vectors = np.reshape(grid, (self._num_point, self._num_dim))
      offset_adj = OFFSET_ADJ
      offset = self._plane.offset
      # Find a plane that achieves the desired impurity
      for _ in range(MAX_ITER):
        pos_arr = np.array([v for v in vectors if self._plane.isGreater(v)])
        neg_arr = np.array([v for v in vectors if self._plane.isLess(v)])
        other_arr = np.array([v for v in vectors if self._plane.isNear(v)])
        trinary = TrinaryClassification(
            pos_arr=pos_arr,
            neg_arr=neg_arr,
            other_arr=other_arr)
        if best_trinary is None:
          best_trinary = trinary
        elif np.abs(self._impurity - trinary.impurity) <= THR_IMPURITY:
          break
        else:
          # Adjust the plane to approach the desired impurity
          offset, offset_adj = adjustOffset(trinary.impurity, offset, offset_adj)
          self._plane = Plane(self._plane.vector, offset)
          if np.abs(trinary.impurity - self._impurity) <  \
              np.abs(best_trinary.impurity - self._impurity):
              best_trinary = trinary
          trinary = None
      if trinary is not None:
        break
    if trinary is None:
      print("**Cannot achieve impurity of %2.2f. Using: %2.2f"
          % (self._impurity, best_trinary.impurity))
      trinary = best_trinary
    return trinary
