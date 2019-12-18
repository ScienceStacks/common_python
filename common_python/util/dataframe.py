"""Utilities for DataFrames"""

import pandas as pd
import numpy as np


def isLessEqual(df1, df2):
  """
  Tests if each value in df1 is less than or equal the
  corresponding value in df2.
  """
  df = df1 - df2
  df_tot = df.applymap(lambda v: v <= 0)
  return df_tot.sum().sum() == df.size
