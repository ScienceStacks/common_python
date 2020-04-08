'''Utilities common to classifiers.'''

import collections
import pandas as pd
import numpy as np


FRACTION = "fraction"
STATE = "state"

# Previous, current, and next state in a time series
# p_dist - # indices to previous state
# n_dist - # indices to next state
AdjacentStates = collections.namedtuple(
    "AdjacentStates", "prv cur nxt p_dist n_dist")


def findAdjacentStates(ser_y, index):
  """
  Finds the states adjacent to the index provided where
  The indices are in time serial order.
  :param pd.Series ser_y:
      index is in time series order
  :param object index: an index in ser_y
  :return AdjacentStates:
  """
  cur_state = ser_y[index]
  def findNewState(indices):
    """
    The first index is the current state.
    Returns np.nan if no next state.
    """
    dist = -1  # Starting with current state
    ser = ser_y.loc[indices]
    for _, state in ser.iteritems():
      dist += 1
      if state != cur_state:
        return state, dist
    return np.nan, np.nan
  #
  indices = ser_y.index.tolist()
  pos = indices.index(index)
  prv_indices = indices[:pos+1]
  nxt_indices = indices[pos:]
  prv_indices.reverse()  # So start after pos
  nxt_state, n_dist = findNewState(nxt_indices)
  prv_state, p_dist = findNewState(prv_indices)
  return AdjacentStates(prv=prv_state, cur=cur_state,
      nxt=nxt_state, p_dist=p_dist, n_dist=n_dist)

def calcStateProbs(ser_y):
  """
  Calculates the probability of each state occurrence.
  :param pd.Series ser_y:
      index: instance
      value: state
  :return pd.Series:
      index: state
      value: float
  """
  df = pd.DataFrame()
  df[STATE] = ser_y
  df[FRACTION] = df[STATE]
  dfg = df.groupby(STATE).count()
  df_result = pd.DataFrame(dfg)
  ser = df_result[FRACTION]/len(df)
  return ser

def aggregatePredictions(df_pred, threshold=0.8):
  """
  Aggregates probabilistic predictions, choosing the
  class with the largest probability, if it exceeds
  the threshold.
  :param pd.DataFrame df_pred:
      columns: class
      rows: instance
      values: float
  :param float threshold:
  :return pd.Series:
      index: instance
      values: class or np.nan if below threshold
  """
  MISSING = -1
  columns = df_pred.columns
  values = []
  df = df_pred.applymap(lambda v: 1 if v >= threshold
      else MISSING)
  for idx, row in df_pred.iterrows():
    row_list = row.tolist()
    pos = row_list.index(max(row_list))
    values.append(columns[pos])
  ser = pd.Series(values, index=df_pred.index)
  ser = ser.apply(lambda v: np.nan if v == MISSING else v)
  return ser
