"""Calculates statistics."""

import numpy as np
import pandas as pd


def filterZeroVarianceRows(df):
  """
  Removes rows that have zero variance.
  :param pd.DataFrame df: numeric rows
  :return pd.DataFrame: same columns as df
  """
  ser_std = df.std(axis=1)
  indices = [i for i in ser_std.index 
      if np.isclose(ser_std[i], 0.0)]
  return df.drop(indices)

def calcLogSL(df):
  """
  Calculates the minus log10 of significance levels
  for values in rows.
  :param pd.DataFrame df: Rows have same units
  :return pd.DataFrame: transformed
  """
  df_T = df.T
  ser_std = df_T.std(axis=1)
  ser_mean = df_T.mean(axis=1)
  df1 = df_T.subtract(ser_mean, axis=0)
  df1 = df_T.divide(ser_std, axis=0)
  df_log = df1.applymap(lambda v:
      -np.log10(1 - stats.norm(0, 1).cdf(v)))
  return df_log
