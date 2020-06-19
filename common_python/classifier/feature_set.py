'''Representation and analysis of features in a classifier.'''

import common_python.constants as cn
from common_python.classifier import util_classifier
from common_python.classifier import feature_analyzer
from common_python.util.persister import Persister

import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn

FEATURE_SEPARATOR = "+"
SORT_FUNCTION = lambda v: float(v[1:])


############### CLASSES ####################
class FeatureSet(object):
  '''Represetation of a set of features'''

  def __init__(self, descriptor, analyzer=None):
    """
    Parameters
    ----------
    descriptor: list/set/string/FeatureSet
    analyzer: FeatureAnalyzer
    """
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

  def __str__(self):
    return self.str

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

  def equals(self, other):
    diff = self.set.symmetric_difference(other.set)
    return len(diff) == 0

  def profile(self, sort_function=SORT_FUNCTION):
    """
    Creates a DataFrame that the feature sets provided
    by calculating the contribution to the classification.
    The profile assumes the use of SVM so that the effect
    of features is additive. Summing the value of 
    features in a
    feature set results in a float. 
    If this is < 0, it is the
    0 class, and if > 0, it is the 1 class.

    Values for an instance are calculated by using 
    that instance as a test set.

    Parameters
    ----------
    sort_function: Function
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
      #coefs = util_classifier.binaryMultiFit(clf,
      #    df_sub, ser_sub)
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
    if sort_function is not None:
      sorted_index = sorted(instances, key=sort_function)
      df = df.reindex(sorted_index)
    return df

  def plotProfile(self, is_plot=True,
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
    df_profile = self.profile()
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
