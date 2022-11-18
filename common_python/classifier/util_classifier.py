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

from common_python import constants as cn

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
#
ClassifierDescription = collections.namedtuple(
    "ClassifierDescription", "clf features")
#
# score - score for cross validations
# scores - list of scores
# clfs - list of classifiers
BinaryCrossValidateResult = collections.namedtuple(
    "BinaryCrossValidateResult",
    "score scores clfs")


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
    for _, state in ser.items():
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

def makeArrayDF(df, indices=None):
  """
  Constructs numpy arrays for the dataframe.
  :param pd.DataFrame df:
  :return ndarray:
  """
  if indices is None:
    indices = df.index
  if isinstance(df, pd.Series):
    df = pd.DataFrame(df)
  return df.loc[indices, :].to_numpy()

def makeArraySer(ser, indices=None):
  """
  Constructs numpy arrays for the dataframe.
  :param pd.Series ser:
  :return ndarray:
  """
  if indices is None:
    indices = ser.index
  return ser.loc[indices].to_numpy()

def makeArrays(df, ser, indices=None):
  """
  Constructs numpy arrays for the dataframe and series.
  :param pd.DataFrame df:
  :param pd.Series ser:
  :return ndarray, ndarray:
  """
  if indices is None:
    indices = df.index
  return makeArrayDF(df, indices=indices),  \
      makeArraySer(ser, indices=indices)

def scoreFeatures(clf, df_X, ser_y,
    features=None, train_idxs=None, test_idxs=None,
    is_copy=True):
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
  :param bool is_copy: Copy the classifier
  :return float: score for classifier using features
  """
  if train_idxs is None:
    train_idxs = df_X.index
  if test_idxs is None:
    test_idxs = df_X.index
  if features is None:
    features = df_X.columns.tolist()
  #
  if is_copy:
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

def getBinarySVMParameters(clf):
  return clf.intercept_[0], clf.coef_[0]

def predictBinarySVM(clf, values):
  """
  Does binary SVM prediction from the SVM coeficients.
  :param SVC clf:
  :param list/array values:
  :return int in [0, 1]:
  """
  intercept, arr1 = getBinarySVMParameters(clf)
  arr2 = np.array(values)
  svm_value = arr1.dot(arr2) + intercept
  if svm_value < 0:
    binary_class = cn.NCLASS
  else:
    binary_class = cn.PCLASS
  return binary_class

def binaryMultiFit(clf, df_X, ser_y,
    list_train_idxs=None):
  """
  Constructs multiple fits for indices for a classifier
  that has an intercept and coefs_.
  :param Classifier clf:
  :param pd.DataFrame df_X:
      columns: features
      index: instances
  :param pd.Series ser_y:
      index: instances
      values: binary class values (0, 1)
  :param list-indexes
  :return float, array
      intercept
      mean values of coeficients:
  """
  if list_train_idxs is None:
    list_train_idxs = [df_X.index.to_list()]
  coefs = np.repeat(0.0, len(df_X.columns))
  intercept = 0
  length = len(list_train_idxs)
  for train_idxs in list_train_idxs:
    arr_X, arr_y = makeArrays(df_X, ser_y,
        indices=train_idxs)
    clf.fit(arr_X, arr_y)
    coefs += clf.coef_[0]
    intercept += clf.intercept_[0]
  return intercept/length, coefs/length

def binaryCrossValidate(clf, df_X, ser_y,
    partitions=None, num_holdouts=1, num_iteration=10):
  """
  Constructs a cross validated estimate of the score
  for a classifier trained on the features in df_X.
  :param Classifier clf:
  :param pd.DataFrame df_X:
      columns: features
      index: instances
  :param pd.Series ser_y:
      index: instances
      values: binary class values (0, 1)
  :param list-(list-index, list-index) partitions:
      list of pairs of indices. Pairs are train, test.
  :param int num_holdouts: holdouts used if
      constructing partitions
  :param int num_iteration: number of iterations in
      cross validations if constructing partitions
  :return BinaryCrossValidate:
  """
  if partitions is None:
    partitions = [p for p in 
        partitioner(ser_y, num_iteration,
        num_holdout=num_holdouts)]
  scores = []
  features = df_X.columns.tolist()
  clfs = []
  for train_set, test_set in partitions:
    copy_clf = copy.deepcopy(clf)
    scores.append(scoreFeatures(copy_clf, df_X, ser_y,
        features=features, is_copy=False,
        train_idxs=train_set, test_idxs=test_set))
    clfs.append(copy_clf)
  result = BinaryCrossValidateResult(
      score=np.mean(scores), scores=scores, clfs=clfs)
  return result

def correlatePredictions(clf_desc1, clf_desc2,
    df_X, ser_y, partitions):
  """
  Estimates the correlation of predicted classifications
  between two classifiers.
  :param ClassifierDescription clf_desc1:
  :param ClassifierDescription clf_desc2:
  :param pd.DataFrame df_X:
      columns: features
      index: instances
  :param pd.Series ser_y:
      index: instances
      values: binary class values (0, 1)
  :param list-(list-index, list-index) partitions:
      list of pairs of indices. Pairs are train, test.
  :return float: correlation of two predictions
  """
  def predict(clf_desc, train_idxs, test_idxs):
    df_X_sub = pd.DataFrame(df_X[clf_desc.features])
    arr_X_train, arr_y_train = makeArrays(
        df_X_sub, ser_y, indices=train_idxs)
    clf_desc.clf.fit(arr_X_train, arr_y_train)
    arr_X_test = makeArrayDF(df_X_sub, indices=test_idxs)
    arr_y_pred = clf_desc.clf.predict(arr_X_test)
    return arr_y_pred
  #
  arr1 = np.array([])
  arr2 = np.array([])
  for train_idxs, test_idxs in partitions:
    yv1 = predict(clf_desc1, train_idxs, test_idxs)
    yv2 = predict(clf_desc2, train_idxs, test_idxs)
    arr1 = np.concatenate([arr1, yv1])
    arr2 = np.concatenate([arr2, yv2])
  is_zero = np.isclose(np.std(arr1), 0) or \
      np.isclose(np.std(arr2), 0)
  if is_zero:
    result = 0.0
  else:
    result = np.corrcoef(arr1, arr2)[0,1]
  if np.isnan(result):
    result = 0.0
  return result

def partitioner(ser, count, num_holdout=1):
  """
  Creates multiple training and test indexes by 
  randomly selecting a indices for a classifier.
  This is an alternative to partitionByState.

    Parameters
    ----------
    ser : pd.Series
      Values: state
      index: instance
    count: int
      Number of partitions
    num_holdout : int, optional
      Number of holdouts per state to form tests.

    Returns
    -------
    :return iterator.
        The iterator returns a pair of
        test and training indices
    """
  classes = ser.unique().tolist()
  classes.sort()
  for _ in range(count):
    test_set = []
    for cls in classes:
      ser_cls = ser[ser == cls]
      if len(ser_cls) <= num_holdout:
        raise ValueError(
            "Class %s has fewer than %d holdouts" %
            (cls, num_holdout))
      idxs = random.sample(
          ser_cls.index.tolist(), num_holdout)
      test_set.extend(idxs)
    #
    train_set = list(set(ser.index).difference(test_set))
    yield train_set, test_set

def makePartitions(**kwargs):
  """
  Creates partitions.

    Parameters
    ----------
    kwargs: dict

    Returns
    -------
    :return list-list-index
    """
  return [p for p in makePartitioner(**kwargs)]

def makePartitioner(partitions=None, num_iteration=None,
    num_holdout=1, ser_y=None):
  """
  Creates a partitioner type iterator for partitions.

    Parameters
    ----------
    partitions: list-list-index
    num_itereation: int
        Number of partitions to create
    num_holdout: int
        Number of holdouts for each state
    ser_y: pd.Series
        Class values

    Returns
    -------
    :return iterator.
        The iterator returns a pair of
        test and training indices
    """
  if partitions is None:
    iterator = partitioner(ser_y, num_iteration,
        num_holdout=num_holdout)
  elif isinstance(partitions, collections.abc.Iterable):
    iterator = partitions
  else:
    raise RuntimeError("Invalid input")
  for train_set, test_set in iterator:
    yield train_set, test_set

def backEliminate(clf, df_X, ser_y, partitions,
    max_decr_score=0.001):
  """
  Uses backwards elimination to remove features without
  a significant reduction in the accuracy score.

  Parameters
  ----------
  clf: Classifier
  df_X: Feature matrix
  ser_y: Calssification vector.
  partitions: list-list-index_pairs
  max_decr_score: float
      Maximum amount by which score can decrease
      when deleting a feature.
  Returns
  -------
  list-object, float.
      Features selected, score for these features
  """
  features = df_X.columns.to_list()
  max_iteration = len(features)
  bcv_result = binaryCrossValidate(clf, df_X, ser_y,
      partitions=partitions)
  last_score = bcv_result.score
  for _ in range(max_iteration):
    is_changed = False
    if len(features) > 1:
      for feature in features:
        new_features = list(features)
        new_features.remove(feature)
        bcv_result = binaryCrossValidate(clf, 
            df_X[new_features],
            ser_y, partitions=partitions)
        new_score = bcv_result.score
        if last_score - new_score <= max_decr_score:
          features.remove(feature)
          is_changed = True
          last_score = new_score
          break
    if not is_changed:
      break
  return features, last_score
