"""Experiments for evaluating classifiers in a grid."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import common_python.constants as cn
from common_python.plots.plotter import Plotter

DEF_NUM_DIM = 2
DEF_VECTOR = np.repeat(1, DEF_NUM_DIM)
DEF_DENSITY = 10
SMALL = 1e-5


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
    # list of positive and negative arrays
    self.pos_arrs, self.neg_arrs, self.other_arrs = self._makeLabels()

  def _makeGrid(self):
    """
    Creates a uniform grid on a space of arbitrary dimension.
    :param float density: points per unit
    """
    coords = [np.linspace(self._min_val, self._max_val,
        self._density*(self._max_val - self._min_val))
        for _ in range(self._num_dim)]
    return np.meshgrid(*coords)

  def _makeLabels(self):
    """
    Creates the labels for the grid based on the classification vector.
    :return N X 2 matrix, N X 2 matrix: positive, negative labelled vectors
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
    return pos_arrs, neg_arrs, other_arrs

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
    
  def plotGrid(self, pos_arrs=None, neg_arrs=None,
      coef_arr=None, is_plot=True):
    """
    Plots classes on a grid.
    :param list-np.array pos_arrs: vector of positive classes
    :param list-np.array neg_arrs: vector of negative classes
    :param np.array coef_arr: vector orthogonal to plan
    :param bool is_plot: do the plot
    """
    plotter = Plotter()
    def plot(vectors, color):
      xv, yv = zip(*vectors)
      plotter.ax.scatter(xv, yv, color=color)
    #
    if pos_arrs is None:
      pos_arrs = self.pos_arrs
    if neg_arrs is None:
      neg_arrs = self.neg_arrs
    if coef_arr is None:
      coef_arr = self._coef_arr
    plot(self.pos_arrs, "blue")
    plot(self.neg_arrs, "red")
    # Add the line for the hyper plane
    x_arr, y_arr = ExperimentHypergrid._makePlotValues(
        coef_arr, self._xlim, self._ylim)
    plotter.ax.plot(x_arr, y_arr, color="black")
    # Do the plot
    plotter.do(title="Hypergrid", xlim=self._xlim,
        ylim=self._ylim, is_plot=is_plot)
