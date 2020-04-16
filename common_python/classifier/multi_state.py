"""
Classifier for multiple states constructed from a
bass classifier. Classifier features
are selected separately for each state (class).

The base classifier must expose methods for fit,
predict, score. Fitting is done by forward selection
of features to achieve a desired accuracy.
This is a "soft" constraint, and it is checked
without the use of cross validation (for performance
reasons).
"""

import common_python.constants as cn
from common_python.util import util
from common_python.classifier import util_classifier

import collections
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm


###################################################
class MultiState(object):
 
  def __init__(self, base_clf=svm.LinearSVC(),
        desired_accuracy=0.9):
    """
    :param Classifier base_clf: 
    :param float desired_accuracy: accuracy for clf
    """
    self.base_clf = base_clf
    self.states = []
    self.all_features = None
    self.feature_dct = {}  # key is state; value is ser
    self.clf_dct = {}  # key is state
    self.score_dct = {}  # key is state

  def fit(self, df_X, ser_y):
    """
    Selects the top features for each state and fits
    a classifier with the desired accuracy.
    Classifiers are in self.clfs.
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: state (class)
    :globals:
        assigned: all_features, states, features_dct, 
                  clf_dct, score_dct
       
    """
    self.states = ser_y.unique()
    self.all_features = df_X.columns.tolist()
    df_fstat = util_classifier.makeFstatDF(df_X, ser_y)
    for state in self.states:
      # Initialize for this state
      self.clf_dct[state] = copy.deepcopy(self.base_clf)
      ser_fstat = df_fstat[state]
      ser_fstat.sort_values()
      all_features = ser_fstat.index.tolist()
      ser_y_state = util_classifier.makeOneStateSer(ser_y,
          state)
      # Use enough features to obtain the desired accuracy
      # This may not be possible
      for rank in range(len(features)):
        self.feature_dct[state] = features_state[0:rank+1]
        df_X_rank = self._setFeatures(df_X, state)
        self.clf_dct[state].fit(df_X_rank, ser_y_rank)
        self.score_dct[state] = clf.score(df_X, ser_y)
        if self.score_dct[state] >= self.desired_accuracy:
          break

  def _setFeatures(df_X, state):
    """
    Initializes the features for a state.
    :return pd.DataFrame: Non-feature columns are 0
    :globals:
        read: all_features
    """
    non_features = list(set(
        self.all_features).difference(
        self.features_dct[state]))
    df_X_sub = df_X.copy()
    df_X_sub[non_features] = 0
    return df_X_sub

  def predict(self, df_X):
    """
    Predict using the results of the classifier for each
    state.
    :param pd.DataFrame: features, indexed by instance.
    :return pd.DataFrame: fraction prediction by state;
        columns are states; index by instance
    Notes:
      1. Assumes that self.clfs has been previously populated 
         with fitted classifiers.
      2. Considers all states encountered during fit
    """

  def score(self, df_X, ser_y):
    """
    Evaluates the accuracy of the ensemble classifier for
    the instances pvodied.
    :param pd.DataFrame df_X: columns are features, rows are instances
    :param pd.Series ser_y: rows are instances, values are states
    Returns a float in [0, 1] which is the accuracy of the
    ensemble classifier.
    Assumes that self.clfs has been previously populated with fitted
    classifiers.
    """
    df_predict = self.predict(df_X)
    missing_columns = set(ser_y).difference(
        df_predict.columns)
    for column in missing_columns:
      df_predict[column] = np.repeat(0,
          len(df_predict))
    accuracies = []
    for instance in ser_y.index:
      # Accuracy is the probability of selecting the correct state
      try:
        accuracy = df_predict.loc[instance, ser_y.loc[instance]]
      except:
        import pdb; pdb.set_trace()
      accuracies.append(accuracy)
    return np.mean(accuracies)

  def _orderFeatures(self, clf, state_selection=None):
    """
    Orders features by descending value of importance.
    :param int state_selection: restrict analysis to a single state
    :return list-int:
    """
    values = self.clf_desc.getImportance(clf,
        state_selection=state_selection)
    length = len(values)
    sorted_tuples = np.argsort(values).tolist()
    # Calculate rank in descending order
    result = [length - sorted_tuples.index(v) for v in range(length)]
    return result

  def makeRankDF(self, state_selection=None):
    """
    Constructs a dataframe of feature ranks for importance,
    where the rank is the feature order of importance.
    A more important feature has a lower rank (closer to 0)
    :param int state_selection: restrict analysis to a single state
    :return pd.DataFrame: columns are cn.MEAN, cn.STD, cn.STERR
    """
    df_values = pd.DataFrame()
    for idx, clf in enumerate(self.clfs):
      df_values[idx] = pd.Series(self._orderFeatures(clf,
          state_selection=state_selection),
          index=self.features)
    df_result = self._makeFeatureDF(df_values)
    df_result = df_result.fillna(0)
    return df_result.sort_values(cn.MEAN)

  def makeImportanceDF(self, state_selection=None):
    """
    Constructs a dataframe of feature importances.
    :param int state_selection: restrict analysis to a single state
    :return pd.DataFrame: columns are cn.MEAN, cn.STD, cn.STERR
    The importance of a feature is the maximum absolute value of its coefficient
    in the collection of classifiers for the multi-state vector (one vs. rest, ovr).
    """
    ABS_MEAN = "abs_mean"
    df_values = pd.DataFrame()
    for idx, clf in enumerate(self.clfs):
      values = self.clf_desc.getImportance(clf,
         state_selection=state_selection)
      df_values[idx] = pd.Series(values, index=self.features)
    df_result = self._makeFeatureDF(df_values)
    df_result[ABS_MEAN] = [np.abs(x) for x in df_result[cn.MEAN]]
    df_result = df_result.sort_values(ABS_MEAN, ascending=False)
    del df_result[ABS_MEAN]
    df_result = df_result.fillna(0)
    return df_result

  def _plot(self, df, top, fig, ax, is_plot, **kwargs):
    """
    Common plotting codes
    :param pd.DataFrame: cn.MEAN, cn.STD, indexed by feature
    :param str ylabel:
    :param int top:
    :param bool is_plot: produce the plot
    :param ax, fig: matplotlib
    :param dict kwargs: keyword arguments for plot
    """
    # Data preparation
    if top == None:
      top = len(df)
    indices = df.index.tolist()
    indices = indices[0:top]
    df = df.loc[indices, :]
    # Plot
    if ax is None:
      fig, ax = plt.subplots()
    ax.bar(indices, df[cn.MEAN], yerr=df[cn.STD],
        align='center', 
        alpha=0.5, ecolor='black', capsize=10)
    bottom = util.getValue(kwargs, "bottom", 0.25)
    plt.gcf().subplots_adjust(bottom=bottom)
    ax.set_xticklabels(indices, rotation=90, fontsize=10)
    ax.set_ylabel(kwargs[cn.PLT_YLABEL])
    ax.set_xlabel(util.getValue(kwargs, cn.PLT_XLABEL, "Gene Group"))
    this_max = max(df[cn.MEAN] + df[cn.STD])*1.1
    this_min = min(df[cn.MEAN] - df[cn.STD])*1.1
    this_min = min(this_min, 0)
    ylim = util.getValue(kwargs, cn.PLT_YLIM,
        [this_min, this_max])
    ax.set_ylim(ylim)
    if cn.PLT_TITLE in kwargs:
      ax.set_title(kwargs[cn.PLT_TITLE])
    if is_plot:
      plt.show()
    return fig, ax
