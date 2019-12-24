"""Experiments for evaluating classifiers in a grid."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import common_python.constants as cn
import common_python.util.util as util
from common_python.plots.plotter import Plotter

DEF_NUM_DIM = 2
DEF_COEF = np.repeat(1, DEF_NUM_DIM)
DEF_OFFSET = 0
DEF_DENSITY = 10
SMALL = 1e-5
POS = 1
NEG = -1
NEG = -1


class Vector(object):
  """
  Representation of a vector
  """

  def __init__(self, coef_arr):
    self.coef_arr = np.array(coef_arr)
    self.dim = len(self.coef_arr)

  def dot(self, vector):
    return self.coef_arr.dot(vector.coef_arr)


class Plane(object):
  """
  Representation of a hyperplane.
  """

  def __init__(self, coef_arr, offset=DEF_OFFSET):
    """
    :param np.array coef_arr: coefficients of the normal to the plane
    :param float offset: offset from 0
    """
    self.coef_arr = coef_arr
    self.offset = offset
    self._coordinates = None

  def makeCoordinates(self, xlim, ylim):
    """
    Construct coordinates for the hyperplane in 2 dimensions.
    coef[0]*x + coef[1]*y - offset = 0; 
    so y = coef[0]*x/coef[1] + offset/coef[1]
    :param tuple-float xlim: lower, upper x values
    :param tuple-float ylim: lower, upper y values
    :return array-float, array-float: x-values, y-values
    """
    if self._coordinates is None:
      x_arr = np.array(xlim)
      slope = -self.coef_arr[0] / self.coef_arr[1]
      offset = self.offset / self.coef_arr[1]
      y_arr = x_arr*slope + self.offset / self.coef_arr[1]
      self._coordinates = x_arr, y_arr
    return self._coordinates

  def isLess(self, coef_arr):
    """
    The vector is at a negative angle to the plane.
    :param float coef_arr: vector whose position is evaluated
    """
    return self.coef_arr.dot(coef_arr) - self.offset < -SMALL 

  def isGreater(self, coef_arr):
    """
    The vector is at a positive angle to the plane.
    :param float coef_arr: vector whose position is evaluated
    """
    return self.coef_arr.dot(coef_arr) - self.offset > SMALL

  def isNear(self, coef_arr):
    """
    The vector is near the plane.
    :param float coef_arr: vector whose position is evaluated
    """
    return np.abs(self.coef_arr.dot(coef_arr) - self.offset) <= SMALL


class TrinaryClassification(object):
  """
  Classification of instances into:
     positive class, negative class, other (fall on boundary)
  """
  
  def __init__(self, 
      pos_arr=None, neg_arr=None, other_arr=None):
    """
    :param list-array pos_arr:
    :param list-array neg_arr:
    :param list-array other_arr: neither pos nor neg
    """
    self.pos_arr = np.array(util.setList(pos_arr))
    self.neg_arr = np.array(util.setList(neg_arr))
    self.other_arr = np.array(util.setList(other_arr))
    self.dim = len(self.pos_arr[0])
    self._df_feature = None
    self._ser_label = None

  @property
  def df_feature(self):
    if self._df_feature is None:
      self._df_feature, self._ser_label = self.makeMatrices()
    return self._df_feature

  @property
  def ser_label(self):
    if self._ser_label is None:
      self._df_feature, self._ser_label = self.makeMatrices()
    return self._ser_label

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
        pos_arr=adjust(self.pos_arr),
        neg_arr=adjust(self.neg_arr),
        other_arr=adjust(self.other_arr))

  def makeMatrices(self):
    """
    Makes feature and label matrices.
    Only includes pos and neg labels.
    :param TrinaryData trinary:
    :return pd.DataFrame, pd.Series:
    """
    df = pd.concat([pd.DataFrame(self.pos_arr),
        pd.DataFrame(self.neg_arr)])
    ser = pd.concat([
        pd.Series(np.repeat(POS, len(self.pos_arr))),
        pd.Series(np.repeat(NEG, len(self.neg_arr))),
        ])
    indices = range(len(df))
    df.index = indices
    ser.index = indices
    return df, ser
    

class ExperimentHypergrid(object):

  def __init__(self, density=DEF_DENSITY,
      num_dim=2, min_val=-1, max_val=1, plane=None):
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
    if plane is None:
      plane = Plane(DEF_COEF, DEF_OFFSET)
    self._plane = plane
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
    pos_arr = np.array([v for v in vectors 
        if self._plane.isGreater(v)])
    neg_arr = np.array([v for v in vectors if self._plane.isLess(v)])
    other_arr = np.array([v for v in vectors
        if self._plane.isNear(v)])
    return TrinaryClassification(
        pos_arr=pos_arr,
        neg_arr=neg_arr,
        other_arr=other_arr)

  def perturb(self, sigma):
    """
    Perturb the ground truth.
    :param float sigma: standard deviation for perturbation
    :return TrinaryClassification:
    """
    return self.trinary.perturb(sigma)
    
  def plotGrid(self, trinary=None, plane=None,
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
    if plane is None:
      plane = self._plane
    plot(trinary.pos_arr, "blue")
    plot(trinary.neg_arr, "red")
    # Add the line for the hyper plane
    x_arr, y_arr = plane.makeCoordinates(self._xlim, self._ylim)
    plotter.ax.plot(x_arr, y_arr, color="black")
    # Do the plot
    plotter.do(title="Hypergrid", xlim=self._xlim,
        ylim=self._ylim, is_plot=is_plot)
