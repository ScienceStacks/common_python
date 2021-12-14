"""Utilities for DataFrames"""

import pandas as pd
import numpy as np


def isLessEqual(df1, df2):
  """
  Tests if each value in df1 is less than or equal the
  corresponding value in df2.
  """
  indices = set(df1.index).intersection(df2.index)
  dff1 = df1.loc[indices, :]
  dff2 = df2.loc[indices, :]
  df = dff1 - dff2
  df_tot = df.applymap(lambda v: v <= 0)
  result = df_tot.sum().sum() == df.size
  return result

def mean(dfs):
  """
  Calculates the mean of values in a list of dataframes
  for the same index, column.
  :param list-pd.DataFrame dfs:
  :return pd.DataFrame:
  """
  df_mean = sum(dfs)
  return df_mean/len(dfs)

def std(dfs):
  """
  Calculates the standard deviation of values in a 
  list of dataframes for the same index, column.
  :param list-pd.DataFrame dfs:
  :return pd.DataFrame:
  """
  df_mean = mean(dfs)
  df_sq = sum([(df - df_mean)*(df - df_mean) for df in dfs])
  return df_sq / len(dfs)

def intersection(df1, df2, axis=1):
  """
  Constructs a dataframe that is df1 for the intersection of the columns (rows).

  Parameters
  ----------
  df1: pd.DataFrame
  df2: pd.DataFrame
  axis: int
      0 - rows
      1 - columns
  
  Returns
  -------
  pd.DataFrame
  """
  if axis == 1:
    columns = list(set(df1.columns).intersection(df2.columns))
    return df1[columns]
  elif axis == 0:
    indices = list(set(df1.index).intersection(df2.index))
    return df1.loc[indices, :]
  else:
    raise ValueError("Invalid axis: %d" % axis)
    
  
