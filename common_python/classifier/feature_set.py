'''Representation and analysis of features in a classifier.'''

import common_python.constants as cn
from common_python.classifier import util_classifier
from common_python.classifier import feature_analyzer
from common_python.util.persister import Persister
from common_python.util import util

import argparse
import copy
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.stats as stats
import seaborn

FEATURE_SEPARATOR = "+"
SORT_FUNC = lambda v: float(v[1:])
PROB_EQUAL = 0.5


############### CLASSES ####################
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
    elif isinstance(descriptor, set)  \
        or isinstance(descriptor, list):
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
    if self._analyzer is not None:
      fit_partitions = [t for t, _ in 
          self._analyzer.partitions]
      self.intercept, self.coefs  \
          = util_classifier.binaryMultiFit(
          self._analyzer._clf,
          self._analyzer.df_X[self.list],
          self._analyzer.ser_y,
          list_train_idxs=fit_partitions)

  def __str__(self):
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

  def evaluate(self, df_X, min_count=3):
    """
    Evaluates feature vector for FeatureSet to assess 
    statistical significance. If no data are present,
    the result is np.nan.

    Parameters
    ----------
    df_X: pd.DataFrame
        Feature vector
    min_count: int
        minimum number of case occurrences

    Returns
    -------
    pd.Series.
      value: signifcance level
      index: instance index from df_X
    """
    df_trinary = self.profileTrinary()
    siglvls = []
    for instance in df_X.index:
      trinary_values = tuple(df_X.loc[
          instance, self.list])
      count = df_trinary.loc[[trinary_values], cn.COUNT]
      count = count.values[0]
      if count < min_count:
        siglvls.append(np.nan)
      else:
        siglvls.append(df_trinary.loc[
            [trinary_values], cn.SIGLVL_POS].values[0])
    ser = pd.Series(siglvls)
    ser.index = df_X.index
    return ser

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
    # FIXME
    #if title is None:
    #  title = "Score: %3.2f" % self.ser_comb.loc[self.str]
    ax.set_title(title)
    ax.legend()
    if is_plot:
      plt.show()
