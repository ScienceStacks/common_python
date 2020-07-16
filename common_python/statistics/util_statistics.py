"""Calculates statistics."""

import itertools
import numpy as np
import pandas as pd
import random
import scipy.stats as stats
import operator as op
from functools import reduce


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

def generalizedBinomialDensity(probs, num_choose,
   is_sampled=False):
  """
  Calculates the probability of exactly n events out of a
  set of size |probs|. This is a generalization of the
  binomial where events can have different
  probabilities.
  :param list-float probs: probability of each event
  :param int num_choose: number of events
  :param bool is_sampled: sample the combinatorics space
  :return float:
  Notes:
    1. The exact calculation scales poorly forr
       len(probs) > 10.
    2. Sampled is fairly accurate if there is not a
       large difference between the probabilities,
       especially if the maximum difference is < 0.1.
  """
  SMALL_PROB = 0.01  # Max prob of not choosing an index
  num_prob = len(probs)
  if num_choose > num_prob:
    msg = "Combination size must be lessr than the list"
    raise ValueError(msg)
  #
  indices = list(range(num_prob))
  set_indices = set(indices)
  prob_arr = np.array(probs)
  result = 0
  if is_sampled:
    # Select the number of samples so that the probability
    # of *not* choosing an index is very small.
    # SMALL_PROB >= (1 - 1/num_prob)**num_sample
    # log10(SMALL_PROB)/log10(1 - 1/num_prob)
    #         <= num_sample
    num_sample = int(np.log10(SMALL_PROB)/
        np.log10(1 - 1/num_prob))
    num_sample = 100*num_sample
    for _ in range(num_sample):
      occur_idxs = random.sample(indices, num_choose)
      non_occur_idxs = list(set_indices.difference(
          occur_idxs))
      occur_prob = prob_arr[occur_idxs].prod()
      non_occur_prob = (1 - prob_arr[
          non_occur_idxs]).prod()
      result += occur_prob*non_occur_prob
    num_combs = choose(len(probs), num_choose)
    frac = num_sample/num_combs
    result = result/frac
  else:
    for combination in itertools.combinations(indices,
        num_choose):
      if (len(combination) < num_prob)  \
           and (len(combination) > 0):
        occurred_indices = np.array(combination)
        prob_occurred = prob_arr[occurred_indices].prod()
        non_occurred_indices = np.array(
            list(set_indices.difference(
            occurred_indices)))
        non_occurreds = 1 - prob_arr[non_occurred_indices]
        prob_non_occurred = non_occurreds.prod()
        prob = prob_occurred*prob_non_occurred
      elif len(combination) == num_prob:
        prob = prob_arr[indices].prod()
      elif len(combination) == 0:
        non_occurreds = 1 - prob_arr[indices]
        prob = non_occurreds.prod()
      result += prob
  return result

def choose(num_total, num_choose):
  r = min(num_choose, num_total - num_choose)
  numer = reduce(op.mul, range(num_total,
      num_total - r, -1), 1)
  denom = reduce(op.mul, range(1, r+1), 1)
  return numer // denom

def generalizedBinomialTail(probs, num_choose,
    **kwargs):
  """
  Calculates P(n >= num_choose)
  :param list-float probs: probability of each event
  :param int num_choose: number of events
  :param dict kwargs: passed to generalizedBionomial
  :return float:
  """
  length = len(probs)
  if num_choose > length:
    msg = "Combination size must be lessr than the list"
    raise ValueError(msg)
  #
  result = 0
  for num in range(num_choose, length+1):
    result += generalizedBinomialDensity(probs, num,
        **kwargs)
  return result
