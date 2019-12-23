"""Experiments for evaluating classifiers in a grid."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import common_python.constants as cn
import common_python.util.util as util
from common_python.plots.plotter import Plotter

DEF_NUM_DIM = 2
DEF_VECTOR = np.repeat(1, DEF_NUM_DIM)
DEF_DENSITY = 10
SMALL = 1e-5


class TrinaryClassification(object):
  """
  Classification of instances into:
     positive class, negative class, other (fall on boundary)
  """
  
  def __init__(self, 
      pos_arrs=None, neg_arrs=None, other_arrs=None):
    """
    :param list-array pos_arrs:
    :param list-array neg_arrs:
    :param list-array other_arrs: neither pos nor neg
    """
    self.pos_arrs = util.setList(pos_arrs)
    self.neg_arrs = util.setList(neg_arrs)
    self.other_arrs = util.setList(other_arrs)
    self.dim = len(self.pos_arrs[0])

  def perturb(self, sigma):
    """
    Adds a N(0, sigma) to each value
    :param float sigma: standard deviation
    :return TrinaryClassification:
    """
    def adjust(arrs):
      return [np.random.normal(0, sigma, self.dim) + v
          for v in arrs]
    #
    return TrinaryClassification(
        pos_arrs=adjust(self.pos_arrs),
        neg_arrs=adjust(self.neg_arrs),
        other_arrs=adjust(self.other_arrs))
    

class ExperimentHypergrid(object):

  def __init__(self, density=DEF_DENSITY,
      num_dim=2, min_val=-1, max_val=1, coef_arr=DEF_VECTOR):
    """
    :param int num_dim: number of dimensions
    :param float density: number of coordinates along a unit
                        distance for an axis
    :param float min_val: minimum value for an axis
    :param float max_val: maximum value for an axis
    :param array coef_arr: vector orthogonal to separating hyperplane
    """
    self._num_dim = num_dim
    self._density = density
    self._min_val = min_val
    self._max_val = max_val
    self._coef_arr= coef_arr
    self._xlim = [self._min_val, self._max_val]
    self._ylim = [self._min_val, self._max_val]
    # Computed
    self.grid = self._makeGrid()
    self.trinary = self._makeTrinary()

  def _makeGrid(self):
    """
    Creates a uniform grid on a space of arbitrary dimension.
    :param float density: points per unit
    """
    coords = [np.linspace(self._min_val, self._max_val,
        self._density*(self._max_val - self._min_val))
        for _ in range(self._num_dim)]
    return np.meshgrid(*coords)

  def _makeTrinary(self):
    """
    Creates the labels for the grid based on the classification vector.
    :return TrinaryClassification:
    """
    num_rows = (self._density  \
        * (self._max_val - self._min_val))**self._num_dim
    coords = [np.reshape(self.grid[n], (num_rows, 1))
        for n in range(len(self.grid))]
    vectors = [np.array([x[0],y[0]])
        for x,y in zip(coords[0], coords[1])]
    pos_arrs = np.array([v for v in vectors
        if np.dot(v, self._coef_arr) > SMALL])
    neg_arrs = np.array([v for v in vectors
        if np.dot(v, self._coef_arr) < -SMALL])
    other_arrs = np.array([v for v in vectors
        if np.abs(np.dot(v, self._coef_arr)) <= SMALL])
    return TrinaryClassification(
        pos_arrs=pos_arrs,
        neg_arrs=neg_arrs,
        other_arrs=other_arrs)

  @staticmethod
  def _makePlotValues(coef_arr, xlim, ylim):
    """
    Construct the x-values and y-values for a plot.
    a*x + b*y = 0; so y = a/b*x
    :param array coef_arr: equation for a line
        (or the coef_arr orthogonal to the line)
    :param tuple-float xlim: lower, upper x values
    :param tuple-float ylim: lower, upper y values
    :return array-float, array-float: x-values, y-values
    """
    x_arr = np.array(xlim)
    factor = -coef_arr[0] / coef_arr[1]
    y_arr = x_arr*factor
    return x_arr, y_arr

  def perturb(self, sigma):
    """
    Perturb the ground truth.
    :param float sigma: standard deviation for perturbation
    :return TrinaryClassification:
    """
    return self.trinary.perturb(sigma)
    
  def plotGrid(self, trinary=None, coef_arr=None,
      is_plot=True):
    """
    Plots classes on a grid.
    :param TrinaryClassification trinary:
    :param np.array coef_arr: vector orthogonal to plan
    :param bool is_plot: do the plot
    """
    plotter = Plotter()
    def plot(vectors, color):
      xv, yv = zip(*vectors)
      plotter.ax.scatter(xv, yv, color=color)
    #
    if trinary is None:
      trinary = self.trinary
    if coef_arr is None:
      coef_arr = self._coef_arr
    plot(trinary.pos_arrs, "blue")
    plot(trinary.neg_arrs, "red")
    # Add the line for the hyper plane
    x_arr, y_arr = ExperimentHypergrid._makePlotValues(
        coef_arr, self._xlim, self._ylim)
    plotter.ax.plot(x_arr, y_arr, color="black")
    # Do the plot
    plotter.do(title="Hypergrid", xlim=self._xlim,
        ylim=self._ylim, is_plot=is_plot)
