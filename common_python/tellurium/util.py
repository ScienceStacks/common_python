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

def calcRsq(ser_obs, ser_est):
  ser_res = ser_obs - ser_est
  return 1 - ser_res.var() / ser_obs.var()

def interpolateTime(ser, time):
  """
  Interpolates a values between two times.
  :param pd.Series ser: index is time
  :param float time:
  :return float:
  """
  def findTime(a_list, func):
    if len(a_list) == 0:
      return np.nan
    else:
      return func(a_list)
  def findValue(time):
    if np.isnan(time):
      return np.nan
    else:
      return ser[time]
  #
  time_lb = findTime([t for t in ser.index if t <= time], max)
  time_ub = findTime([t for t in ser.index if t >= time], min)
  value_lb = findValue(time_lb)
  value_ub = findValue(time_ub)
  if np.isnan(value_lb):
    return value_ub
  if np.isnan(value_ub):
    return value_lb
  if time_ub == time_lb:
    return value_ub
  frac = (time - time_lb)/(time_ub - time_lb)
  return (1 - frac)*value_lb + frac*value_ub
