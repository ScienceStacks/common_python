"""Generates Data from an Empirical Distribution."""


import pandas as pd
import numpy as np

import common_python.constants as cn
from common_python.plots import util_plots


class EmpiricalDistributionGenerator(object):
  """
  DataFrames are structured so that columns are features and
  rows are instances (observations).
  Useage:
    1. Simple data.
       generator = EmpiricalDistributionGenerator(df)
       df_data = generator.sample(nobs)
    2. Data with replacing values.
       generator = EmpiricalDistributionGenerator(df)
       df_data = generator.synthesize(nobs, frac)
  """

  def __init__(self, df):
    """
    :param pd.DataFrame: empirical distribution
    """
    self._df = df

  def sample(self, nobs, is_decorrelate=True):
    """
    Samples with replacement.
    :param int nobs:
    """
    df_sample = self._df.sample(nobs, replace=True)
    if is_decorrelate:
      df_sample = self.__class__.decorrelate(df_sample)
    df_sample = df_sample.reset_index()
    return df_sample

  @staticmethod
  def decorrelate(df):
    """
    Permutes rows within columns to remove correlations between features.
    :return pd.DataFrame:
    """
    length = len(df)
    df_result = df.copy()
    for col in df_result.columns:
      values = df_result[col].tolist()
      df_result[col] = np.random.permutation(values)
    return df_result

  def synthesize(self, nobs, frac, **kwargs):
    """
    Returns a random sample of rows with frac values replaced
    by values from the empirical CDF.
    :param int nobs: number of observations in the sample
    :param float frac: fraction of values replaced
    :param dict kwargs: arguments passed to sampler
    :return pd.DataFrame:
    """
    return
