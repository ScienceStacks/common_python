"""Experiments for evaluating classifiers in a grid."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import common_python.constants as cn
from common_python.plots.plotter import Plotter

DEF_NUM_DIM = 2
DEF_VECTOR = np.repeat(1, DEF_NUM_DIM)
DEF_DENSITY = 10


class ExperimentHypergrid(object):

  def __init__(self, density=DEF_DENSITY,
      num_dim=2, min_val=-1, max_val=1, classification_vec=DEF_VECTOR):
    """
    :param int num_dim: number of dimensions
    :param float density: number of coordinates along a unit
                        distance for an axis
    :param float min_val: minimum value for an axis
    :param float max_val: maximum value for an axis
    :param array classification_vec: vector orthogonal to separating hyperplane
    """
    self._num_dim = num_dim
    self._density = density
    self._min_val = min_val
    self._max_val = max_val
    self._classification_vec= classification_vec
    # Computed
    self.grid = self._makeGrid()
    self.pos_vecs, self.neg_vecs = self._makeLabels()

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
    pos_vecs = np.array([v for v in vectors
        if np.dot(v, self._classification_vec) > 0])
    neg_vecs = np.array([v for v in vectors
        if np.dot(v, self._classification_vec) < 0])
    return pos_vecs, neg_vecs

  def _makePlotValues(self, vector, xlim, ylim):
    """
    Construct the x-values and y-values for a plot.
    a*x + b*y = 0; so y = a/b*x
    :param array vector: equation for a line
        (or the vector orthogonal to the line)
    :param tuple-float xlim: lower, upper x values
    :param tuple-float ylim: lower, upper y values
    :return array-float, array-float: x-values, y-values
    """
    xv = np.array(xlim)
    factor = -vector[0] / vector[1]
    yv = xv*factor
    return xv, yv
    
  def plotGrid(self, pos_vecs=None, neg_vecs=None,
      fitted_plane=None, is_plot=True):
    """
    Plots classes on a grid.
    :param list-np.array pos_vecs: vector of positive classes
    :param list-np.array neg_vecs: vector of negative classes
    :param np.array fitted_plane: vector orthogonal to plan
    :param bool is_plot: do the plot
    """
    plotter = Plotter()
    def plot(vectors, color):
      xv, yv = zip(*vectors)
      plotter.ax.scatter(xv, yv, color=color)
    #
    if pos_vecs is None:
      pos_vecs = self.pos_vecs
    if neg_vecs is None:
      neg_vecs = self.neg_vecs
    plot(self.pos_vecs, "blue")
    plot(self.neg_vecs, "red")
    # Add the line for the hyper plane
    # Do the plot
    plotter.do(title="Hypergrid", is_plot=is_plot)
