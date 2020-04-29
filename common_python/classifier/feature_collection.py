'''Creates a collection of features. Classifier agnostic.
'''

"""
A FeatureCollection implements the following:
  add() - add a feature to features
  remove() - remove a feature from features
  choose() - choose a feature to add
A FeatureCollection exposes a list of features
  features
"""

from common_python.classifier import util_classifier

import copy
import numpy as np
import pandas as pd

MAX_CORR = 0.5  # Maxium correlation with an existing feature
CLASSES = [0, 1]


################### Base Class #################
class FeatureCollection(object):

  def __init__(self, df_X, ser_y, **kwargs):
    """
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: binary class values: 0, 1
    """
    ###### PRIVATE ####
    self._df_X = df_X
    self._ser_y = ser_y
    ###### PUBLIC ####
    # Features ordered by descending value
    self.all = self._order()
    self.chosens = []  # Features chosen
    self.removes = []  # Features rejected

  def _order(self, ser_weight=None):
    """
    Constructs features ordered in descending priority
    :param pd.Series ser_weight: weights applied to instances
    :return list-object: features
    """
    ser_fstat = util_classifier.makeFstatDF(
        self._df_X, self._ser_y,
        ser_weight=ser_weight)[1]
    ser_fstat = ser_fstat.fillna(0)
    ser_fstat.sort_values(ascending=False)
    return ser_fstat.index.tolist()

  def add(self, feature=None, **kwargs):
    """
    Adds a feature for the class selecting
    the top feature not yet chosen.
    :param object feature: specific feature to add
    :return bool: True if a feature was added.
    """
    if feature is None:
      feature = self.choose(**kwargs)
    #
    if feature is None:
      return False
    else:
      self.chosens.append(feature)
      return True

  def choose(self, ordereds=None, **kwargs):
    """
    Chooses a feature to add from
    those not chosen already.
    The default implementation chooses features
    by descending value of F-Statistic.
    :param list-object ordereds: ordered features
    :return object: feature to add
    Should be overridden is in subclasses
    """
    candidates = self.getCandidates(ordereds=ordereds)
    if len(candidates) > 0:
      feature = candidates[0]
    else:
      feature = None
    return feature

  def getCandidates(self, ordereds=None):
    """
    Get the candidate features in order.
    :param list-object ordereds:
    :return list-object:
    """
    if ordereds is None:
      ordereds = self.all
    excludes = set(self.chosens).union(self.removes)
    return [f for f in ordereds if not f in excludes]

  def remove(self, feature=None):
    """
    Removes a specified feature, or the
    last one added if none is specified.
    :param object feature:
    """
    if feature is None:
      feature = self.chosens[-1]
    self.removes.append(feature)
    self.chosens.remove(feature)


########### Select based on correations #################
class FeatureCollectionCorr(FeatureCollection):
  """
  Selects features for a class using correlations.
  """

  def __init__(self, df_X, ser_y, max_corr=MAX_CORR):
    """
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: class
    :param float max_corr: maximum correlation
        between a new feature an an existing feature
    """
    super().__init__(df_X, ser_y)
    # Private
    self._max_corr = max_corr
    self._df_corr = self._df_X.corr()
    self._df_corr = self._df_corr.fillna(0)

  def choose(self, **kwargs):
    """
    Finds feature to add for class.
    :return bool: True if a feature was added.
    """
    if len(self.chosens) > 0:
      # Not the first feature
      # Correlate chosen features with those not considered
      # Columns are chosens; rows are new candidates
      df_corr = copy.deepcopy(self._df_corr)
      df_corr = pd.DataFrame(
          df_corr.loc[:, self.chosens])
      candidates = self.getCandidates()
      df_corr = df_corr.loc[candidates, :]
      # Find candidates with sufficiently low correlation
      # with chosens
      df_corr = df_corr.applymap(lambda v: np.abs(v))
      ser_max = df_corr.max(axis=1)
      candidates = ser_max[
          ser_max < self._max_corr].index.tolist()
      # Choose the highest priority feature that is
      # not highly correlated with the existing features.
      features = [f for f in self.all if f in candidates]
    else:
      # Handle first feature
      features = self.all
    if len(features) > 0:
      feature = features[0]
    else:
      feature = None
    return feature


####### Select based on classification residuals #######
class FeatureCollectionResidual(FeatureCollection):
  """
  Selects features by using a residual idea for classification.
  The residual is the set of misclassified instances.
  Extra weight is used for these instances.
  """

  def __init__(self, df_X, ser_y, weight=None):
    """
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: class
    :param float max_corr: maximum correlation
        between a new feature an an existing feature
    :param float weight: weight given to misclassifications
        default is 1 + ratio of non-misses to misses
    """
    super().__init__(df_X, ser_y)
    self._weight = weight

  def choose(self, ser_pred=None):
    """
    Finds feature to add.
    :param pd.Series ser_pred: predicted class
    :return object: feature
    """
    indices_miss = self._ser_y.index[
        self._ser_y != ser_pred]
    # Construct the weight for each instance
    ser_weight = self._ser_y.copy()
    ser_weight[:] = 1
    if self._weight is None:
      weight = 1 + (len(self._ser_y)  \
      - len(indices_miss)) / len(indices_miss)
    else:
      weight - self._weight
    ser_weight.loc[indices_miss] = weight
    #
    ordereds = self._order(ser_weight=ser_weight)
    return super().choose(ordereds=ordereds)
