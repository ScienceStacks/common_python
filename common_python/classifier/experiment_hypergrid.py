"""Experiments for evaluating classifiers in a grid."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DEF_NUM_DIM = 2
DEF_VECTOR = np.repeat(1, DEF_NUM_DIM)
DEF_DENSITY = 10


class ExperimentHypergrid(object):

  def __init__(self, density=DEF_DENSITY,
      num_dim=2, min_val=-1, max_val=1, classification_vector=DEF_VECTOR):
    """
    :param int num_dim: number of dimensions
    :param float density: number of coordinates along a unit
                        distance for an axis
    :param float min_val: minimum value for an axis
    :param float max_val: maximum value for an axis
    :param array classification_vector: vector orthogonal to separating hyperplane
    """
    self._num_dim = num_dim
    self._density = density
    self._min_val = min_val
    self._max_val = max_val
    self._classification_vector= classification_vector
    # Computed
    self.grid = self._makeGrid()
    self.pos_vectors, self.neg_vectors = self._makeLabels()

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
    num_rows = self._density(self._max_val - self._min_val)**self._num_dim
    import pdb; pdb.set_trace()
    coords = [np.reshape(self.grid[n], num_rows, 1)
        for n in len(self.grid)]
    vectors = [np.array(v) for v in zip(*coords)]
    pos_vectors = np.array([v for v in vectors if v*self._classification_vector > 0])
    neg_vectors = np.array([v for v in vectors if v*self._classification_vector <= 0])
    return pos_vectors, neg_vectors
    
  def plotGrid(self, pos_vectors, neg_vectors, fitted_plane=None):
    """
    Plots classes on a grid.
    """
    #plt.scatter(

