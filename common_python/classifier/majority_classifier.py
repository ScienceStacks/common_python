"""
Implements a majority class (label) classifier that
always predicts the most frequently occurring class/label.
"""

import numpy as np
import pandas as pd


class MajorityClassifier(object):

  def __init__(self):
    self.majority = None

  def _check(self):
    if self.majority is None:
      raise ValueError("Must fit classifier before using predict or score.")

  def __repr__(self):
    return "Label: %s" % str(self.majority)

  def fit(self, _, ser_label):
    """
    Fit for a majority class classifier. This classifier is used
    in the calculation of a relative score.
    :param pd.Series ser_class:
        index: instance
        value: class label
    Updates
      self.majority_class - most frequnetly occurring class
    """
    ser = ser_label.value_counts()
    self.majority = ser.index[0]

  def predict(self, df_feature):
    """
    :param pd.DataFrame df_feature:
        index: instance
        column: feature
    :return pd.Series: 
        index: instance
        value: predicted class label
    """
    self._check()
    return pd.Series(np.repeat(self.majority, len(df_feature)))

  def score(self, _, ser_label):
    """
    Scores for a majority class classifier
    :param pd.Series ser_label:
        index: instance
        value: class label
    :return float:
    """
    self._check()
    num_match = len([c for c in ser_label if c == self.majority])
    return num_match / len(ser_label)
