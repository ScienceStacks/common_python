'''Utilities common to classifiers.'''

"""
Functions know about features and states.
The feature matrix df_X has columns that are
feature names, and index that are instances.
Trinary values are in the set {-1, 0, 1}.
The state Series ser_y has values that are
states, and indexes that are instances.
The state F-statistic for a gene quantifies
how well it distinguishes between states.
"""

import collections
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats


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
  state with the largest probability, if it exceeds
  the threshold.
  :param pd.DataFrame df_pred:
      columns: state
      rows: instance
      values: float
  :param float threshold:
  :return pd.Series:
      index: instance
      values: state or np.nan if below threshold
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

def makeFstatSer(df_X, ser_y,
     is_prune=True, state_equ=None):
  """
  Constructs the state F-static for gene features.
  This statistic quantifies the variation of the
  gene expression between states to that within
  states.
  :param pd.DataFrame df_X:
      column: gene
      row: instance
      value: trinary
  :param pd.Series ser_y:
      row: instance
      value: state
  :param bo0l is_prune: removes nan and inf values
  :param dict state_equ: Provides for state equivalences
      key: state in ser_y
      value: new state
  """
  if state_equ is None:
      state_equ = {s: s for s in ser_y.unique()}
  # Construct the groups
  df_X = df_X.copy()
  df_X[STATE] = ser_y.apply(lambda v: state_equ[v])
  # Calculate aggregations
  dfg = df_X.groupby(STATE)
  dfg_std = dfg.std()
  dfg_mean = dfg.mean()
  dfg_count = dfg.count()
  # Calculate SSB
  ser_mean = df_X.mean()
  del ser_mean[STATE]
  df_ssb = dfg_count*(dfg_mean - ser_mean)**2
  ser_ssb = df_ssb.sum()
  # SSW
  df_ssw = dfg_std*(dfg_count - 1)
  ser_ssw = df_ssw.sum()
  # Calculate the F-Statistic
  ser_fstat = ser_ssb/ser_ssw
  ser_fstat = ser_fstat.sort_values(ascending=False)
  if is_prune:
    ser_fstat = ser_fstat[ser_fstat != np.inf]
    sel = ser_fstat.isnull()
    ser_fstat = ser_fstat[~sel]
  return ser_fstat

def plotStateFstat(state, df_X, ser_y, is_plot=True):
  """
  Plot state F-statistic
  :param pd.DataFrame df_X:
      column: gene
      row: instance
      value: trinary
  :param pd.Series ser_y:
      row: instance
      value: state
  :param bool is_plot: Construct the plot
  """
  if state is None:
      state_equ = {s: s for s in ser_y.unique()}
  else:
      state_equ = {s: s if s==state else -1 for s in ser_y.unique()}
  num_state = len(state_equ.values())
  ser_fstat = makeFstatSer(df_X, ser_y,
      state_equ=state_equ)
  ser_sl = ser_fstat.apply(lambda v: -np.log(
      1 - scipy.stats.f.cdf(v, num_state-1,
      len(df_X)-num_state)))
  indices = ser_sl.index[0:10]
  _ = plt.figure(figsize=(8, 6))
  _ = plt.bar(indices, ser_sl[indices])
  _ = plt.xticks(indices, indices, rotation=90)
  _ = plt.ylabel("-log(SL)")
  if state is None:
      _ = plt.title("All States")
  else:
      _ = plt.title("State: %d" % state)
  if state is not None:
      _ = plt.ylim([0, 1.4])
  if is_plot:
    plt.show()
