"""Utilities for DataFrames"""

import pandas as pd
import numpy as np


def isLessEqual(df1, df2):
  """
  Tests if each value in df1 is less than or equal the
  corresponding value in df2.
  """
  indices = list(set(df1.index).intersection(df2.index))
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

def subset(df, items, axis=1):
  """
  Constructs a dataframe is a subset to the items, by row or column.

  Parameters
  ----------
  df: pd.DataFrame
  items: list
      columns if axis = 1
      indices if axis = 0
  axis: int
      0 - rows
      1 - columns
  
  Returns
  -------
  pd.DataFrame
  """
  if axis == 1:
    columns = list(set(items).intersection(df.columns))
    return df[columns]
  elif axis == 0:
    indices = list(set(items).intersection(df.index))
    return df.loc[indices, :]
  else:
    raise ValueError("Invalid axis: %d" % axis)
    
  
