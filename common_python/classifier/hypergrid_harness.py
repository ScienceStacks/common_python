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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Place holder
class Vector(object):
  def __init__(self, _):
    pass

DEF_NUM_DIM = 2
DEF_ARR = np.repeat(1, DEF_NUM_DIM)
DEF_OFFSET = 0
DEF_DENSITY = 10
SMALL = 1e-5
POS = 1
NEG = -1
NEG = -1


##################### FUNCTIONS ###################
def _assignObjectValues(target, source):
  """
  Assigns the values in the source to the target.
  Source and target must be the same type.
  :param object target:
  :param object source:
  """
  for key, value in source.__dict__.iteritems():
    target.__dict__[key] = value


#################### CLASSES ######################
class Vector(object):
  """
  Representation of a vector
  """

  def __init__(self, coef_arr):
    self.coef_arr = np.array(coef_arr)
    self.dim_int = len(self.coef_arr)

  def dot(self, vector):
    return self.coef_arr.dot(vector.coef_arr)

  def toArray(vector):
    if isinstance(vector, Vector):
      return vector.coef_arr
    else:
      return vector

  def __str__(self):
    stg = ""
    for idx in range(self.dim_int):
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
  def dim_int(self):
    return self.vector.dim_int

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
    self.dim_int = len(self.pos_arr[0])
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

  def perturb(self, sigma=0, repl_int=1):
    """
    Adds a N(0, sigma) to each value
    :param float sigma: standard deviation
    :param int repl_int: number of replications
    :return list-TrinaryClassification:
    """
    def adjust(arrs):
      return [np.random.normal(0, sigma, self.dim_int) + v
          for v in arrs]
    #
    trinarys = []
    for _ in range(repl_int):
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
    dim_int = trinary.dim_int
    for tri in trinarys[1:]:
      if tri.dim_int != dim_int:
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

  def __init__(self, density=DEF_DENSITY,
      min_val=-1, max_val=1, plane=None):
    """
    :param float density: number of coordinates along a unit
                        distance for an axis
    :param float min_val: minimum value for an axis
    :param float max_val: maximum value for an axis
    :param Plane plane: separating hyperplane
    """
    self._density = density
    self._min_val = min_val
    self._max_val = max_val
    if plane is None:
      plane = Plane(Vector(DEF_ARR), DEF_OFFSET)
    self._plane = plane
    self._xlim = [self._min_val, self._max_val]
    self._ylim = [self._min_val, self._max_val]
    # Computed
    self.grid = self._makeGrid()
    self.trinary = self._makeTrinary()

  @classmethod
  def initImpure(cls, density=DEF_DENSITY, impurity=0):
    """
    Creates a harness with the desired impurity, the difference
    between the fraction of positive and negative instances.
    :param float density: number of coordinates along a unit
                        distance for an axis
    :param float impurity: in [-1, 1]
    :return HypergridHarness:
    """
    raise RuntimeError("Not implemented.")

  @property
  def dim_int(self):
    return self._plane.dim_int

  def _makeGrid(self):
    """
    Creates a uniform grid on a space of arbitrary dimension.
    :param float density: points per unit
    """
    coords = [np.linspace(self._min_val, self._max_val,
        self._density*(self._max_val - self._min_val))
        for _ in range(self.dim_int)]
    # The grid is structured as:
    #  coordinate (e.g., x, y)
    #  row
    #  value
    grid = np.meshgrid(*coords)
    return grid

  def _makeTrinary(self):
    """
    Creates the labels for the grid based on the classification vector.
    :return TrinaryClassification:
    """
    rows_int = (self._density  \
        * (self._max_val - self._min_val))**self.dim_int
    coords = np.reshape(self.grid, (self.dim_int, rows_int))
    coords = np.transpose(coords)
    #coords = [np.reshape(self.grid[n], (rows_int, 1))
    #    for n in range(len(self.grid))]
    # TODO: Generalize so works for n-dimensions
    pos_arr = np.array([v for v in coords
        if self._plane.isGreater(v)])
    neg_arr = np.array([v for v in coords
        if self._plane.isLess(v)])
    other_arr = np.array([v for v in coords
        if self._plane.isNear(v)])
    return TrinaryClassification(
        pos_arr=pos_arr,
        neg_arr=neg_arr,
        other_arr=other_arr)

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
    plane.plot(plotter, self._xlim, self._ylim, color="black")
    # Do the plot
    if not cn.PLT_TITLE in kwargs:
      title = "Separating Hyperplane: %s" % str(self._plane)
    plotter.do(title=title, xlim=self._xlim,
        ylim=self._ylim, is_plot=is_plot)
    #
    return plotter

  def evaluateSVM(self, mclf=None, sigma=0, repl_int=1, is_plot=True):
    """
    Evaluates the classification accuracy of an svm.
    Assumes dim_int === 2.
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
      trinarys = self.trinary.perturb(sigma=sigma, repl_int=repl_int)
      dfs_feature = [trinary.df_feature for trinary in trinarys]
      ser_label = trinarys[0].ser_label
      cv_results = mclf.cross_validate(dfs_feature, ser_label)
    # Plot construction. 
    if is_plot:
      if self.dim_int != 2:
        raise ValueError("Must have a 2-dimensional grid")
      mclf.fit(trinary.df_feature, trinary.ser_label)
      vector = Vector(mclf.coef_)
      offset = -mclf.intercept_ / (mclf.coef_[0][0])
      plane = Plane(vector, offset=offset)
      # TODO: Do the plot
    #
    return np.mean(cv_result['test_score']), plane
    #return mclf.score(trinary.df_feature, trinary.ser_label), plane


class HypergridHarnessMetaClassifier(HypergridHarness):

  # Values are dataframes with the columns cn.MEAN, cn.STD
  # and rows are MetaClassifiers evaluated.
  ScoreResults = collections.namedtuple("ScoreResults",
      "abs rel")

  def __init__(self, mclfs, density=DEF_DENSITY, impurity=0):
    """
    :param list-MetaClassifier mclfs: to be studied
    :param float density: number of coordinates along a unit
                        distance for an axis
    :param float impurity: in [-1, 1]
    """
    self.mclfs = mclfs
    harness = HypergridHarness.initImpure(density, impurity)
    # Copy all data to the new to this harness
    assignObjectValues(self, harness)

  def _evaluateExperiment(self, sigma=0, repl_int=1):
    """
    Evaluates the classification accuracy of MetaClassifiers
    for a single experiment.
    :param float sigma:
    :param int repl_int: Number of replications passed to classifiers
    :return list-ScoreResult:
    """
    train_trinarys = self.trinary.perturb(sigma=sigma, repl_int=repl_int)
    test_trinary = self.trinary.perturb(sigma=sigma, repl_int=1)[0]
    dfs = [trinary.df_feature for trinary in train_trinarys]
    [m.fit(dfs, self.trinary.ser_label) for m in selfmclfs]
    score_results = [
        m.score(test_trinary.df_feature, self.trinary.ser_label)
        for m in self.mclfs]
    return score_results

  def evaluate(self, count=10, sigmas=[0], repl_ints=[1]):
    """
    Evaluates the classification accuracy of MetaClassifier
    for different conditions. Each experiment is repeated several times.
    :param list-float sigmas:
    :param list-int repl_ints:
    :return dict:
         key: (sigma,repl_int)
         value: ScoreResults
    """
    #
    result = {}
    for sigma in sigmas:
      for repl_int in repl_ints:
        aggregator_abs = ItemAggregator(lambda s: s.abs)
        aggregator_rel = ItemAggregator(lambda s: s.rel)
        for _ in range(count):
          key = (sigma, repl_int)
          results = self._evaluateExperiment(sigma=sigma,
              repl_int=repl_int)
          aggregator_abs.append(results)
          aggregator_rel.append(results)
        #
        df_abs = aggregator_abs.df
        df_rel = aggregator_rel.df
        result[(sigma, repl_int)] = ScorerResults(
            abs=df_abs, rel=df_rel)
    #
    return result
