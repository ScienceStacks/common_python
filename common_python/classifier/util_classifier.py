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
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
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

def makeOneStateSer(ser, state):
  """
  Creates a Series where state is 1 for the designated
  state and 0 otherwise.
  :param pd.Series ser:
       index: instance
       value: int (state)
  :param int
  :return pd.Series
  """
  result = ser.map(lambda v: 1 if v==state else 0)
  return result

def makeFstatDF(df_X, ser_y, ser_weight=None):
  """
  Constructs the state F-static for gene features
  by state.
  :param pd.DataFrame df_X:
      column: gene
      row: instance
      value: trinary
  :param pd.Series ser_y:
      row: instance
      value: state
  :param pd.Series ser_weight: weight for instances
      row: instance
      value: multiplier for instance
  :return pd.DataFrame:
     columns: state
     index: gene
     value: -log significance level
     ordered by descending magnitude of sum(value)
  """
  MAX = "max"
  if ser_weight is None:
    ser_weight = ser_y.copy()
    ser_weight.loc[:] = 1
  df_X_adj = df_X.copy()
  df_X_adj = df_X_adj.apply(lambda col: col*ser_weight)
  states = ser_y.unique()
  df = pd.DataFrame()
  for state in states:
    ser_y_1 = makeOneStateSer(ser_y, state)
    ser = makeFstatSer(df_X_adj, ser_y_1, is_prune=False)
    df[state] = ser
  df[MAX] = df.max(axis=1)
  df = df.sort_values(MAX, ascending=False)
  del df[MAX]
  return df

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
      state_equ = {s: s if s==state else -1 for s 
          in ser_y.unique()}
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

def plotInstancePredictions(ser_y, ser_pred,
    is_plot=True):
  """
  Plots the predicted states with text codings of
  the actual states.
  :param pd.Series ser_y: actual states
       States are numeric
  :param pd.Series ser_pred: actual states
  :param bool is_plot: Produce the plot
  """
  min_state = ser_y.min()
  max_state = ser_y.max()
  plt.figure(figsize=(12, 8))
  plt.scatter([-1,80], [-1,7])
  for obs in range(len(ser_y)):
      index = ser_pred.index[obs]
      _ = plt.text(obs, ser_pred[index],
          "%d" % ser_y[index], fontsize=16)
  plt.xlim([0, len(ser_y)])
  plt.ylim([-0.5+min_state, 0.5+max_state])
  _ = plt.xlabel("Observation", fontsize=18)
  _ = plt.ylabel("Predicted State", fontsize=18)
  if is_plot:
    plt.show()

def makeArrays(df, ser, indices=None):
  """
  Constructs numpy arrays for the dataframe and series.
  :param pd.DataFrame df:
  :param pd.Series ser:
  :return ndarray, ndarray:
  """
  if indices is None:
    indices = df.index
  return df.loc[indices, :].values,  \
      ser.loc[indices].values

def scoreFeatures(clf, df_X, ser_y,
    features=None, train_idxs=None, test_idxs=None):
  """
  Evaluates the classifier for the set of features and the
  training and test indices provided (or all if None are
  provided).
  :param Classifier clf: Exposes
      fit, score
  :param pd.DataFrame df_X:
      columns: features
      indicies: instances
  :param pd.Series ser_y:
      indices: instances
      values: classes
  :param list-object features:
  :param list-object train_idxs: indices for training
  :param list-object test_idxs: indices for testing
  :return float: score for classifier using features
  """
  if train_idxs is None:
    train_idxs = df_X.index
  if test_idxs is None:
    test_idxs = df_X.index
  if features is None:
    features = df_X.columns.tolist()
  #
  clf = copy.deepcopy(clf)
  arr_X, arr_y = makeArrays(df_X[features], ser_y,
      indices=train_idxs)
  clf.fit(arr_X, arr_y)
  #
  arr_X, arr_y = makeArrays(df_X[features], ser_y,
      test_idxs)
  score = clf.score(arr_X, arr_y)
  #
  return score

def partitionByState(ser, holdouts=1):
  """
  Creates training and test indexes by randomly selecting
  a indices for each state.
  :param pd.DataFrame ser: Classes for instances
  :param int holdouts: number of holdouts for test
  :return list-object, list-object: test, train
  """
  classes = ser.unique().tolist()
  classes.sort()
  test_idxs = []
  for cls in classes:
    ser_cls = ser[ser == cls]
    if len(ser_cls) <= holdouts:
      raise ValueError(
          "Class %s has fewer than %d holdouts" %
          (cls, holdouts))
    idxs = random.sample(ser_cls.index.tolist(),
        holdouts)
    test_idxs.extend(idxs)
  #
  train_idxs = list(set(ser.index).difference(test_idxs))
  return train_idxs, test_idxs
  return test_idxs
