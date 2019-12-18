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
