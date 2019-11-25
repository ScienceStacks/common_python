'''Utilities used in Tellurium.'''

import pandas as pd
import numpy as np

def dfToSer(df):
  """
  Converts a dataframe to a series.
  :param pd.DataFrame df:
  :return pd.Series:
  """
  return pd.concat([df[c] for c in df.columns])

def isEqualList(list1, list2):
  return len(set(list1).symmetric_difference(list2)) == 0

def readFile(path):
  with open(path, "r") as fd:
    result = fd.readlines()
  return "\n".join(result)
