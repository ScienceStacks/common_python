"""
Experimental harness for evaluating classifiers in a multidimensional
grid (hypergrid). Ground truth is a set of equally spaced
observations in the grid with a separating hyperplane that
determines positive (pos) and negative (neg) values. Values
on the separating plane are "other".
"""

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
DEF_NUM_DIM = 2
DEF_ARR = np.repeat(1, DEF_NUM_DIM)
DEF_OFFSET = 0
DEF_DENSITY = 10

# Constants
POS = 1
NEG = -1
SMALL = 1e-5

# Parameters
MAX_ITER = 100
THR_IMPURITY = 0.05


#################### CLASSES ######################
class Vector(object):
  """
  Representation of a vector
  """

  def __init__(self, coef_arr):
    """
    :param np.array coef_arr:
    """
    self.coef_arr = np.array(coef_arr)
    self.num_dim = len(self.coef_arr)

  def dot(self, vector):
    return self.coef_arr.dot(vector.coef_arr)

  def toArray(vector):
    if isinstance(vector, Vector):
      return vector.coef_arr
    else:
      return vector

  def __str__(self):
    stg = ""
    for idx in range(self.num_dim):
      coef = self.coef_arr[idx]
      if len(stg) == 0:
        stg = "%2.4f*x%d" % (coef, idx+1)
      else:
        stg = "%s + %2.4f*x%d" % (stg, coef, idx+1)
    return stg


class Plane(object):
  """
  Representation of a hyperplane.
  """

  def __init__(self, vector, offset=DEF_OFFSET):
    """
    :param Vector vector: vector normal to the plane
    :param float offset: offset from 0
    """
    self.vector = vector
    self.offset = offset

  def __str__(self):
    return "%s + %2.3f = 0" % (str(self.vector), -self.offset)

  @property
  def num_dim(self):
    return self.vector.num_dim

  def makeCoordinates(self, xlim, ylim):
    """
    Construct coordinates for the hyperplane in 2 dimensions.
    coef[0]*x + coef[1]*y - offset = 0; 
    so y = coef[0]*x/coef[1] + offset/coef[1]
    :param tuple-float xlim: lower, upper x values
    :param tuple-float ylim: lower, upper y values
    :return array-float, array-float: x-values, y-values
    """
    x_arr = np.array(xlim)
    slope = -self.vector.coef_arr[0] / self.vector.coef_arr[1]
    offset = self.offset / self.vector.coef_arr[1]
    y_arr = x_arr*slope + self.offset / self.vector.coef_arr[1]
    return x_arr, y_arr

  def isLess(self, vector):
    """
    The vector is at a negative angle to the plane.
    :param Vector vector: vector whose position is evaluated
    """
    arr = Vector.toArray(vector)
    return self.vector.coef_arr.dot(arr) - self.offset < -SMALL 

  def isGreater(self, vector):
    """
    The vector is at a positive angle to the plane.
    :param Vector vector: vector whose position is evaluated
    """
    arr = Vector.toArray(vector)
    result = self.vector.coef_arr.dot(arr) - self.offset > SMALL
    return result

  def isNear(self, vector):
    """
    The vector is near the plane.
    :param Vector vector: vector whose position is evaluated
    """
    arr = Vector.toArray(vector)
    return np.abs(
        self.vector.coef_arr.dot(arr) - self.offset) <= SMALL

  def plot(self, plotter, xlim, ylim, **kwargs):
    """
    For a two dimensional space, plots the line corresponding to
    the hyperplane.
    :param Plotter plotter:
    :param dict kwargs: optional arguments for plot
    """
    if not cn.PLT_COLOR in kwargs:
      kwargs[cn.PLT_COLOR] = "black"
    x_arr, y_arr = self.makeCoordinates(xlim, ylim)
    plotter.ax.plot(x_arr, y_arr, **kwargs)
 

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
    if len(self.pos_arr) > 0:
      self.num_dim = len(self.pos_arr[0])
    else:
      self.num_dim = len(self.neg_arr[0])
    self.num_point = self._makeNumPoint()
    self.impurity = self._makeImpurity()
    self._df_feature = None
    self._ser_label = None

  def _makeImpurity(self):
    """
    1: all positives
    0: positives == negatives
    -1: all negatives
    """
    num_pos = len(self.pos_arr)
    num_neg = len(self.neg_arr)
    return (num_pos - num_neg)/self.num_point

  def _makeNumPoint(self):
    num_pos = len(self.pos_arr)
    num_neg = len(self.neg_arr)
    return num_pos + num_neg

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

  def perturb(self, sigma=0, num_repl=1):
    """
    Adds a N(0, sigma) to each value
    :param float sigma: standard deviation
    :param int num_repl: number of replications
    :return list-TrinaryClassification:
    """
    def adjust(arrs):
      return [np.random.normal(0, sigma, self.num_dim) + v
          for v in arrs]
    #
    trinarys = []
    for _ in range(num_repl):
      trinarys.append(TrinaryClassification(
          pos_arr=adjust(self.pos_arr),
          neg_arr=adjust(self.neg_arr),
          other_arr=adjust(self.other_arr)))
    return trinarys

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

  @classmethod
  def concat(cls, trinarys):
    """
    Concatenates TrinaryClassifications
    :param list-TrinaryClassification trinarys:
    :return TrinaryClassification:
    """
    trinary = copy.deepcopy(trinarys[0])
    num_dim = trinary.num_dim
    for tri in trinarys[1:]:
      if tri.num_dim != num_dim:
        raise ValueError(
            "All TrinaryClassification must have the same dimension")
      trinary.pos_arr = np.concatenate([trinary.pos_arr,
          tri.pos_arr])
      trinary.neg_arr = np.concatenate([trinary.neg_arr,
          tri.neg_arr])
      trinary.other_arr = np.concatenate([trinary.other_arr,
          tri.other_arr])
    return trinary

class HypergridHarness(object):

  def __init__(self, density=DEF_DENSITY, num_point=None,
      min_val=-1, max_val=1, plane=None, impurity=0.0):
    """
    :param float density: number of coordinates along a unit
                        distance for an axis
    :param int num_point: number of points in grid
                          if None, then all in cross product
    :param float min_val: minimum value for an axis
    :param float max_val: maximum value for an axis
    :param Plane plane: separating hyperplane
    :param float impurity: in [-1, 1] - difference between pos & neg
    """
    self._density = density
    self._min_val = min_val
    self._max_val = max_val
    self._num_point = num_point
    self._impurity = impurity
    if plane is None:
      plane = Plane(Vector(DEF_ARR), DEF_OFFSET)
    self._plane = plane
    self._xlim = [self._min_val, self._max_val]
    self._ylim = [self._min_val, self._max_val]
    # Computed
    self.grid, self.trinary = self._makeGridAndTrinary()

  @classmethod
  def initImpure(cls, impurity=0, **kwargs):
    """
    Creates a harness with the desired impurity, the difference
    between the fraction of positive and negative instances.
    :param float impurity: in [-1, 1]
    :param dict kwargs: arguments in HypergridHarness constructor
    :return HypergridHarness:
    """
    raise RuntimeError("Not implemented.")

  @property
  def num_dim(self):
    return self._plane.num_dim

  def _makeGrid(self):
    """
    Creates a uniform grid on a space of arbitrary dimension.
    :param float density: points per unit
    """
    num = max(2, int(self._density*(self._max_val - self._min_val)))
    coords = [np.random.permutation(
        np.linspace(self._min_val, self._max_val, num))
        for _ in range(self.num_dim)]
    # The grid is structured as:
    #  coordinate (e.g., x, y)
    #  row
    #  value
    if self._num_point is None:
      grid = [g for g in itertools.product(*coords)]
    else:
      grid = [list(g) for n,g in enumerate(itertools.product(*coords))
          if n < self._num_point]
    return grid

  def _makeGridAndTrinary(self):
    """
    Creates the grid and the 
    labels for the grid based on the classification vector.
    :return list-list-float, TrinaryClassification:
    """
    best_impurity = 1
    for _ in range(MAX_ITER):
      grid = self._makeGrid()
      num_row = (np.size(grid)) / self.num_dim
      if not np.isclose(num_row, int(num_row)):
        raise RuntimeError("num_row should be an integer.")
      num_row = int(num_row)
      vectors = np.reshape(grid, (num_row, self.num_dim))
      pos_arr = np.array([v for v in vectors
          if self._plane.isGreater(v)])
      neg_arr = np.array([v for v in vectors
          if self._plane.isLess(v)])
      other_arr = np.array([v for v in vectors
          if self._plane.isNear(v)])
      trinary = TrinaryClassification(
          pos_arr=pos_arr,
          neg_arr=neg_arr,
          other_arr=other_arr)
      if np.abs(self._impurity - trinary.impurity) <= THR_IMPURITY:
        break
      else:
        best_impurity = min(best_impurity, np.abs(trinary.impurity))
        trinary = None
    if trinary is None:
      raise ValueError("Cannot achieve impurity of %2.2f. Best: %2.2f"
          % (self._impurity, best_impurity))
    return grid, trinary

  def perturb(self, **kwargs):
    """
    Perturb the ground truth.
    :param dict kwargs: arguments for Trinary.perturb
    :return list-TrinaryClassification:
    """
    return self.trinary.perturb(**kwargs)
    
  def plotGrid(self, trinary=None, plane=None,
      is_plot=True, **kwargs):
    """
    Plots classes on a grid.
    :param TrinaryClassification trinary:
    :param np.array vector: vector orthogonal to plan
    :param bool is_plot: do the plot
    :param dict kwargs: options for plotter
    :return Plotter:
    """
    # Handle defaults
    if not cn.PLT_TITLE in kwargs:
      title = "Separating Hyperplane: %s" % str(self._plane)
    if not cn.PLT_XLIM in kwargs:
      kwargs[cn.PLT_XLIM]=self._xlim
    if not cn.PLT_YLIM in kwargs:
      kwargs[cn.PLT_YLIM]=self._ylim
    #
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
    plane.plot(plotter, kwargs[cn.PLT_XLIM],
        kwargs[cn.PLT_YLIM], color="black")
    # Do the plot
    plotter.do(title=title, xlim=kwargs[cn.PLT_XLIM],
        ylim=kwargs[cn.PLT_YLIM], is_plot=is_plot)
    #
    return plotter

  def evaluateSVM(self, mclf=None, sigma=0, num_repl=1, is_plot=True):
    """
    Evaluates the classification accuracy of an svm.
    :param MetaClassifier mclf:
    :param float sigma:
    :return float, plane: accuracy measure, separating hyperplane
    """
    trinary = self.trinary.perturb(sigma=sigma)[0]
    if mclf is None:
      mclf = sklearn.svm.LinearSVC()
      cv_result = sklearn.model_selection.cross_validate(
          mclf, trinary.df_feature, trinary.ser_label, cv=3)
      #mclf.fit(trinary.df_feature, trinary.ser_label)
    else:
      trinarys = self.trinary.perturb(sigma=sigma, num_repl=num_repl)
      dfs_feature = [trinary.df_feature for trinary in trinarys]
      ser_label = trinarys[0].ser_label
      cv_results = mclf.cross_validate(dfs_feature, ser_label)
    # Plot construction. 
    if is_plot:
      if self.num_dim != 2:
        raise ValueError("Must have a 2-dimensional grid")
      mclf.fit(trinary.df_feature, trinary.ser_label)
      vector = Vector(mclf.coef_)
      offset = -mclf.intercept_ / (mclf.coef_[0][0])
      plane = Plane(vector, offset=offset)
      # TODO: Do the plot
    #
    return np.mean(cv_result['test_score']), plane
    #return mclf.score(trinary.df_feature, trinary.ser_label), plane
