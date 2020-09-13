"""
Representation and analysis of features in a classifier.
Key concepts are:

  A case is a measurement with a known class label.
  
  Features have trinary values for expression levels: -1, 0, 1.
  
  A feature set is a set of features used for a classifier.

  A feature vector is the assignment of values to a features
  in a feature set.

  The indexed cases for a feature vector are the cases that
  are compatible with the feature vector.
  
  A classifier construted from a feature set partitions the cases.
  That is, two feature vectors for the feature set, their
  indexed cases have a non-null intersection only if the
  feature vectors are identical.
  
  An inferred classification is the majority class of cases
  selected by a feature vector for a feature set.
"""

import common_python.constants as cn
from common_python.classifier import util_classifier
from common_python.classifier import feature_analyzer
from common_python.util.persister import Persister
from common_python.util import util

import argparse
import collections
import copy
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.stats as stats
import seaborn

SORT_FUNC = lambda v: float(v[1:])
PROB_EQUAL = 0.5
MIN_SL = 10e-10
SEPARATORS = ["[", "]"]  # Separators for index values
MIN_COUNT = 3  # Minimum number of cases selected to
               # consider the classification inferred
               # by a case index.


############### CLASSES ####################
class FeatureVector(object):
  """
  Representation of a feature vector for a feature set.
  Provides information about the features and their values.
  """

  def __init__(self, fset: FeatureSet, descriptor):
    """
    Parameters
    ----------
    fset: FeatureSet
    descriptor: tuple/list/Series/FeatureVector/dict/array
    """
    if isinstance(descriptor, dict):
      self.dict = descriptor
    elif isinstance(descriptor, tuple) or  \
        isinstance(descriptor, list) or \
        isinstance(descriptor, np.ndarray):
      self.dict = {f: t for f, t in 
          zip(fset.list, descriptor)}
    elif isinstance(descriptor, pd.Series):
      self.dict = descriptor.to_dict()
    elif isinstance(descriptor, FeatureVector):
      self.dict = descriptor.dict
    else:
      raise RuntimeError(
          "descriptor is invalid type.")
    self.fset = FeatureSet(list(self.dict.keys()))
    self.list = [v for v in self.dict.values()]
    self.tuple = tuple(self.list)

  def __repr__(self):
    stgs = ["%s%s%d%s" % (f, SEPARATORS[0], 
        v, SEPARATORS[1]) for f, v in
        zip(self.fset.list, self.list)]
    return cn.FEATURE_SEPARATOR.join(stgs)

  def equals(self, other):
    if not self.fset.equals(other.fset):
      return False
    result = all([self.dict[k] == other.dict[k]
        for k in self.dict.keys()])
    return result

  @classmethod
  def make(cls, stg: string):
    """
    Creates an FeatureVector object from its string representation.

    Parameters
    ----------
    stg: str
        String representation of case object

    Returns
    ----------
    FeatureVector
    """
    elements = stg.split(cn.FEATURE_SEPARATOR)
    features = []
    values = []
    for element in elements:
      lpos = element.index(SEPARATORS[0])
      features.append(element[:lpos])
      rpos = element.index(SEPARATORS[1])
      values.append(int(element[lpos+1: rpos]))
    fset = FeatureSet(features)
    return FeatureVector(fset, values)


####################################
class FeatureSet(object):
  """
  Represetation of a set of features. Assumes
  an SVM classifier so that the classifier parameters
  are an intercept and feature multipliers.
  """

  def __init__(self, descriptor, analyzer=None):
    """
    Parameters
    ----------
    descriptor: list/set/string/FeatureSet
    analyzer: FeatureAnalyzer
    """
    # The following are values averaged across instances
    self.intercept = None  # y-axis offset
    self.coefs = None  # Feature multipliers
    if isinstance(descriptor, str):
      # Ensure that string is correctly ordered
      self.set = FeatureSet._unmakeStr(descriptor)
      self.str = FeatureSet._makeStr(self.set)
      self._analyzer = analyzer
    elif isinstance(descriptor, set) or  \
        isinstance(descriptor, collections.abc.Iterable):
      self.set = set(descriptor)
      self.str = FeatureSet._makeStr(self.set)
      self._analyzer = analyzer
    elif isinstance(descriptor, FeatureSet):
      self.set = descriptor.set
      self.str = descriptor.str
      self._analyzer = descriptor._analyzer
    else:
      raise ValueError("Invalid argument type")
    self.list = list(self.set)
    self.list.sort()
    if self._analyzer is not None:
      fit_partitions = [t for t, _ in 
          self._analyzer.partitions]
      self.intercept, self.coefs  \
          = util_classifier.binaryMultiFit(
          self._analyzer._clf,
          self._analyzer.df_X[self.list],
          self._analyzer.ser_y,
          list_train_idxs=fit_partitions)

  def __repr__(self):
    return self.str

  def equals(self, other):
    diff = self.set.symmetric_difference(other.set)
    return len(diff) == 0

  @staticmethod
  def _makeStr(features):
    """
    Creates a string for a feature set
    :param iterable-str features:
    :return str:
    """
    f_list = list(features)
    f_list.sort()
    return cn.FEATURE_SEPARATOR.join(f_list)

  @staticmethod
  def _unmakeStr(fset_stg):
    """
    Recovers a feature set from a string
    """
    return set(fset_stg.split(cn.FEATURE_SEPARATOR))

  def profileTrinary(self):
    """
    Profiles the FeatureSet by trinary value of features.

    Parameters
    ----------
    sort_func: Function
        single value; returns float

    Returns
    -------
    pd.DataFrame
        Columns: 
          cn.PREDICTED - predicted class
          cn.FRAC- Fraction of +1 class for feature values
          cn.SIGLVL_POS - significance level at least 
              that count
              in the feature vector for a positive class
          cn.SIGLVL_NEG - significance level at least 
              that count
              in the feature vector for a positive class
          cn.COUNT - number of feature vectors with value
          cn.FEATURE_SET
        Values: contribution to class
        Index: trinary values of features
        Index.name: string representation of FeatureSet.
    """
    def calc(df, is_mean=True):
      dfg = df.groupby(self.list)
      if is_mean:
        ser_count = dfg.mean()
      else:
        ser_count = dfg.count()
      ser_count.index = ser_count.index.tolist()
      return ser_count[ser_count.columns.tolist()[0]]
    #
    df_X = self._analyzer.df_X
    dct = {cn.PREDICTED: [], cn.VALUE: []}
    args = [cn.TRINARY_VALUES for _ in
        range(len(self.list))]
    iterator = itertools.product(*args)
    for feature_values in iterator:
      value = np.array(feature_values).dot(
          self.coefs) + self.intercept
      dct[cn.VALUE].append(feature_values)
      dct[cn.PREDICTED].append(
          util.makeBinaryClass(value))
    df = pd.DataFrame(dct)
    df = df.set_index(cn.VALUE)
    # Counts
    ser_count = calc(df_X, is_mean=False)
    df[cn.COUNT] = ser_count
    # Mean class values
    df_y = pd.DataFrame(self._analyzer.ser_y)
    df_y = df_X[self.list].copy()
    df_y[cn.CLASS] = self._analyzer.ser_y
    ser_fracpos = calc(df_y)
    df[cn.FRAC] = ser_fracpos
    # Nan counts are zeros
    df[cn.COUNT] = df[cn.COUNT].apply(lambda v:
        0 if np.isnan(v) else v)
    # Calculate significance levels
    pos_list = []
    neg_list = []
    for _, row in df.iterrows():
      if row[cn.COUNT] == 0:
        siglvl_pos = np.nan
        siglvl_neg = np.nan
      else:
        count_pos = int(row[cn.COUNT]*row[cn.FRAC])
        if count_pos == 0:
          siglvl_pos = 1.0
          siglvl_neg = 0
        elif count_pos == row[cn.COUNT]:
          siglvl_pos = 0
          siglvl_neg = 1.0
        else:
          siglvl_pos = 1 - stats.binom.cdf(
              count_pos - 1, row[cn.COUNT], PROB_EQUAL)
          siglvl_neg = stats.binom.cdf(
              count_pos, row[cn.COUNT], PROB_EQUAL)
      pos_list.append(siglvl_pos)
      neg_list.append(siglvl_neg)
    df[cn.SIGLVL_POS] = pos_list
    df[cn.SIGLVL_NEG] = neg_list
    df[cn.FEATURE_SET] = self.str
    return df

  def evaluate(self, df_X, min_count=MIN_COUNT,
      is_include_neg=True, min_sl=MIN_SL):
    """
    Evaluates feature vector for FeatureSet to assess 
    statistical significance. Optionally, the significance
    level of the negative class is returned if its
    absolute value is less than that of the positive
    class.  If no data are present, the result is np.nan.

    Parameters
    ----------
    df_X: pd.DataFrame
        Feature vector
    min_count: int
        minimum number of case occurrences
    min_sl: float
        minimum significance level reported

    Returns
    -------
    pd.DataFrame
      cn.SIGLVL: signifcance level
      cn.CASE: string representation of case
      index: instance index from df_X
    """
    df_trinary = self.profileTrinary()
    siglvls = []
    cases = []
    for instance in df_X.index:
      ser = df_X.loc[instance, :]
      case = self.getFeatureVector(ser)
      cases.append(str(case))
      sel = [i == case.tuple for i in 
          df_trinary.index.tolist()]
      df_trinary_values = df_trinary[sel]
      ser = df_trinary_values.T
      ser = pd.Series(ser[ser.columns.tolist()[0]])
      count = df_trinary.loc[sel][cn.COUNT].values[0]
      if count < min_count:
        siglvls.append(1)
      else:
        siglvl_pos = max(min_sl, ser.loc[cn.SIGLVL_POS])
        siglvl_neg = max(min_sl, ser.loc[cn.SIGLVL_NEG])
        if siglvl_pos < siglvl_neg:
            siglvl = siglvl_pos
        else:
            siglvl = - siglvl_neg
        if not is_include_neg:
            siglvl = siglvl_pos
        siglvls.append(siglvl)
    df = pd.DataFrame({
        cn.SIGLVL: siglvls,
        cn.CASE: cases,
        })
    df.index = df_X.index
    import pdb; pdb.set_trace()
    return df

  def profileInstance(self, sort_func=SORT_FUNC):
    """
    Profiles the FeatureSet over instances
    by calculating the contribution to the classification.
    The profile assumes the use of SVM so that the effect
    of features is additive. Summing the value of 
    features in a
    feature set results in a float. 
    If this is < 0, it is the
    0 class, and if > 0, it is the 1 class.

    Values for an instance are calculated by using 
    that instance as a test set. So, the parameters
    obtained may differ from self.intercept, self.coefs.

    Parameters
    ----------
    sort_func: Function
        single value; returns float

    Returns
    -------
    pd.DataFrame
        Columns: 
          features - column for each feature in fset
          cn.INTERCEPT
          cn.SUM - sum of the values of the features
                   and the intercept
          cn.PREDICTED - predicted class
          cn.CLASS - true class value
        Values: contribution to class
        Index: instance, sorted by time and then
    """
    if self._analyzer is None:
      raise RuntimeError("Improperly constructed instance.")
    # Initializations
    dct = {f: [] for f in self.set}
    dct[cn.PREDICTED] = []
    dct[cn.INTERCEPT] = []
    clf = copy.deepcopy(self._analyzer._clf)
    df_X = self._analyzer.df_X[self.list].copy(deep=True)
    ser_y = self._analyzer.ser_y.copy(deep=True)
    # Construct contributions of instances
    instances = df_X.index.tolist()
    predicts = []
    for instance in instances:
      # Create indices for multi-fit
      indices  = ser_y.index[ser_y.index != instance]
      ser_sub = ser_y.loc[indices]
      df_sub = df_X.loc[indices]
      clf.fit(df_sub, ser_sub)
      score = clf.score([df_X.loc[instance, :]],
          [ser_y.loc[instance]])
      predicted = util_classifier.predictBinarySVM(
          clf, df_X.loc[instance, :])
      intercept, coefs = util_classifier.  \
          getBinarySVMParameters(clf)
      for idx, feature in enumerate(self.list):
        dct[feature].append(coefs[idx]  \
            * df_X.loc[instance, feature])
      dct[cn.PREDICTED].append(predicted)
      dct[cn.INTERCEPT].append(intercept)
    df = pd.DataFrame(dct)
    df.index = instances
    df[cn.CLASS] = ser_y
    # Compute the sums
    sers = [df[f] for f in self.list]
    df[cn.SUM] = sum(sers) + df[cn.INTERCEPT]
    if sort_func is not None:
      sorted_index = sorted(instances, key=sort_func)
      df = df.reindex(sorted_index)
    return df

  def plotProfileInstance(self, is_plot=True,
      ylim=[-4, 4], title=None, ax=None, x_spacing=3):
    """
    Constructs a bar of feature contibutions to
    classification.

    Parameters
    ----------
    descriptor: FeatureSet/str
    is_plot: bool
    ylim: list-float
    title: str
    x_spacing: int
        spacing between xaxis labels

    Returns
    -------
    None.
    """
    df_profile = self.profileInstance()
    instances = df_profile.index.to_list()
    columns = [cn.INTERCEPT]
    columns.extend(self.list)
    df_plot = pd.DataFrame(df_profile[columns])
    #
    def shade(mult):
      """
      Shades the region for class 1.
      :param float mult: direction of shading
      """
      class_instances = self._analyzer.ser_y[
          self._analyzer.ser_y == 1].index.tolist()
      values = np.repeat(mult, len(class_instances))
      ax.bar(class_instances, values, alpha=0.3,
          width=1.0, color="grey")
    #
    # Construct the plot
    if ax is None:
      ax = df_plot.plot.bar(stacked=True)
    else:
      df_plot.plot.bar(stacked=True, ax=ax)
    ax.scatter(instances, df_profile[cn.SUM],
        color="red")
    ax.plot([instances[0], instances[-1]], [0, 0],
        color="black")
    shade(ylim[0])
    shade(ylim[1])
    column_values = [df_plot[c].tolist()
        for c in columns]
    labels = [l if i % x_spacing == 0 else "" for i, l
        in enumerate(instances)]
    ax.set_xticklabels(labels)
    #ax.set_ylabel('distance')
    #ax.set_xlabel('instance')
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.legend()
    if is_plot:
      plt.show()

  def getFeatureVector(self, ser_X):
    """
    Extracts the case from the instance, where
    the case is a valid feature set in ser_X.
    This requires handling merged features.
    :param pd.Series ser_X:
    :param FeatureSet fset:
    :return tuple:
          values are in [-1, 0, 1]
          values are ordered by self.list
    """
    dct = {}
    for feature in self.list:
      if feature_analyzer.SEPARATOR in feature:
        # Handle composite feature
        values = []
        features = feature.split(
            feature_analyzer.SEPARATOR) 
        # Accomulate each value
        for fea in features:
          values.append(ser_X.loc[fea])
        counter = collections.Counter(values)
        dct[feature] = counter.most_common(1)[0][0]
      else:
        dct[feature] = ser_X.loc[feature]
    return FeatureVector(self, dct)
