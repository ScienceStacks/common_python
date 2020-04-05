"""Calculates statistics."""

import itertools
import numpy as np
import pandas as pd
import scipy.stats as stats


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

def calcLogSL(df, round_decimal=4, is_nan=True):
  """
  Calculates the minus log10 of significance levels
  for large values in rows.
  :param pd.DataFrame df: Rows have same units
  :param int round_decimal: How many decimal points to round
  :param bool is_nan: force std = 0 to be np.nan
  :return pd.DataFrame: transformed
  """
  ser_std = df.std(axis=1)
  ser_mean = df.mean(axis=1)
  df1 = df.subtract(ser_mean, axis=0)
  df2 = df1.divide(ser_std, axis=0)
  # Calculate minus log10 of significance levels
  # For efficiency reasons, this is done by dictionary lookup
  df3 = df2.fillna(-4.0)
  df3 = df3.applymap(lambda v: np.round(v, 4))
  values = []
  [ [values.append(v) for v in df3[c]] for c in df3.columns]
  vals_lookup = {v: -np.log10(1 - stats.norm(0, 1).cdf(v))
      for v in pd.Series(values).unique()}
  df4 = df3.applymap(lambda v: vals_lookup[v])
  if is_nan:
    df_log = df4.applymap(lambda v: v if v > 0.01 else np.nan)
  else:
    df_log = df4
  return df_log

def decorrelate(df):
  """
  Permutes rows within columns to remove 
      correlations between features.
  :return pd.DataFrame:
  """
  length = len(df)
  df_result = df.copy()
  for col in df_result.columns:
    values = df_result[col].tolist()
    df_result[col] = np.random.permutation(values)
  return df_result

def generalizedBinomialDensity(probs, num_choose):
  """
  Calculates the probability of exactly n events out of a
  set of size |probs|. This is a generalization of the
  binomial where events can have different
  probabilities.
  :param list-float probs: probability of each event
  :param int num_choose: number of events
  :return float:
  """
  length = len(probs)
  if num_choose > length:
    msg = "Combination size must be lessr than the list"
    raise ValueError(msg)
  #
  indices = list(range(length))
  set_indices = set(indices)
  prob_arr = np.array(probs)
  result = 0
  for combination in itertools.combinations(indices,
      num_choose):
    if (len(combination) < length) and (len(combination) > 0):
      occurred_indices = np.array(combination)
      prob_occurred = prob_arr[occurred_indices].prod()
      non_occurred_indices = np.array(
          list(set_indices.difference(occurred_indices)))
      non_occurreds = 1 - prob_arr[non_occurred_indices]
      prob_non_occurred = non_occurreds.prod()
      prob = prob_occurred*prob_non_occurred
    elif len(combination) == length:
      prob = prob_arr[indices].prod()
    elif len(combination) == 0:
      non_occurreds = 1 - prob_arr[indices]
      prob = non_occurreds.prod()
    result += prob
  return result

def generalizedBinomialTail(probs, num_choose):
  """
  Calculates P(n >= num_choose)
  :param list-float probs: probability of each event
  :param int num_choose: number of events
  :return float:
  """
  length = len(probs)
  if num_choose > length:
    msg = "Combination size must be lessr than the list"
    raise ValueError(msg)
  #
  result = 0
  for num in range(num_choose, length+1):
    result += generalizedBinomialDensity(probs, num)
  return result
