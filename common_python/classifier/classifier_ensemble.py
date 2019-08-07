"""Classification done by an ensemble of classifiers."""

import common_python.constants as cn
from common_python.util import util
from common_python.classifier.classifier_collection  \
    import ClassifierCollection

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

RF_ESTIMATORS = "n_estimators"
RF_MAX_FEATURES = "max_features"
RF_BOOTSTRAP = "bootstrap"
RF_DEFAULTS = {
    RF_ESTIMATORS: 500,
    RF_BOOTSTRAP: True,
    }


class ClassifierEnsemble(ClassifierCollection):

  def _plot(self, df, ylabel, top, fig, ax, is_plot, **kwargs):
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
    ax.bar(indices, df[cn.MEAN], yerr=df[cn.STD], align='center', 
        alpha=0.5, ecolor='black', capsize=10)
    bottom = util.getValue(kwargs, "bottom", 0.25)
    plt.gcf().subplots_adjust(bottom=bottom)
    ax.set_xticklabels(indices, rotation=90, fontsize=10)
    ax.set_xlabel('Gene Group')
    ax.set_ylabel(ylabel)
    this_max = max(df[cn.MEAN] + df[cn.STD])*1.1
    this_min = min(df[cn.MEAN] - df[cn.STD])*1.1
    this_min = min(this_min, 0)
    ylim = util.getValue(kwargs, cn.PLT_YLIM, [this_min, this_max])
    ax.set_ylim(ylim)
    if cn.PLT_TITLE in kwargs:
      ax.set_title(kwargs[cn.PLT_TITLE])
    if is_plot:
      plt.show()
    return fig, ax

  def plotRank(self, top=None, fig=None, ax=None, 
      is_plot=True, **kwargs):
    """
    Plots the rank of features for the top valued features.
    :param int top:
    :param bool is_plot: produce the plot
    :param ax, fig: matplotlib
    :param dict kwargs: keyword arguments for plot
    """
    # Data preparation
    df = self.makeRankDF()
    self._plot(df, "Rank", top, fig, ax, is_plot, **kwargs)

  def plotImportance(self, top=None, fig=None, ax=None, 
      is_plot=True, **kwargs):
    """
    Plots the rank of features for the top valued features.
    :param int top:
    :param bool is_plot: produce the plot
    :param ax, fig: matplotlib
    :param dict kwargs: keyword arguments for plot
    """
    df = self.makeImportanceDF()
    self._plot(df, "Importance", top, fig, ax, is_plot, **kwargs)

  def _makeFeatureDF(self, df_values):
    """
    Constructs a dataframe summarizing values by feature.
    :param pd.DataFrame df_values: indexed by feature, columns are instances
    :return pd.DataFrame: columns are cn.MEAN, cn.STD, cn.STERR
        indexed by feature
    """
    df_result = pd.DataFrame({
        cn.MEAN: df_values.mean(axis=1),
        cn.STD: df_values.std(axis=1),
        })
    df_result[cn.STERR] = df_result[cn.STD] / np.sqrt(len(self.clfs))
    return df_result

  # FIXME: Needs test
  def predict(self, df_X):
    """
    Default prediction algorithm. Reports probability of each class.
    The probability of a class is the fraction of classifiers that
    predict that class.
    :param pd.DataFrame: features, indexed by instance.
    :return pd.DataFrame:  probability by class;
        columns are classes; index by instance
    """
    # Change to an array of array of features
    DUMMY_COLUMN = "dummy_column"
    array = df_X.values
    array = array.reshape(len(df_X), -1) 
    # Create a dataframe of class predictions
    clf_predictions = [clf.predict(df_X) for clf in self.clfs]
    instance_predictions = [dict(collections.Counter(x)) for x in zip(*clf_predictions)]
    df = pd.DataFrame()
    df[DUMMY_COLUMN] = np.repeat(-1, len(self.classes))
    for idx, instance in enumerate(instance_predictions):
        ser = pd.Series([x for x in instance.values()], index=instance.keys())
        df[idx] = ser
    del df[DUMMY_COLUMN]
    df = df.applymap(lambda v: 0 if np.isnan(v) else v)
    df = df / len(self.clfs)
    return df.T

  # FIXME: Needs test
  def score(self, df_X, ser_y):
    """
    Returns a float in [0, 1] which is the accuracy of the classifier
    """
    ser = self.predict(df_X)
    accuracies = []
    for instance in ser_y.index:
      # Accuracy is the probability of selecting the correct class
      accuracy = df_X.loc[instance, ser_y.loc[instance]
      accuracies.append(1 - df_X.loc[instance, ser_y.loc[instance])
    return np.mean(accuracies)
    
  
##################################################################### 
class RandomForestEnsemble(ClassifierEnsemble):

  def __init__(self, df_X, ser_y, iterations=5, **kwargs):
    """
    :param pd.DataFrame df_X:
    :param pd.Series ser_y:
    :param int iterations: Number of random forests created
        to obtain a distribution
    :param dict kwargs: arguments passed to classifier
    """
    self.df_X = df_X
    self.ser_y = ser_y
    self.iterations = iterations
    self.features = df_X.columns.tolist()
    self.classes = ser_y.values.tolist()
    adjusted_kwargs = dict(kwargs)
    for key in RF_DEFAULTS.keys():
      if not key in adjusted_kwargs:
        adjusted_kwargs[key] = RF_DEFAULTS[key]
    if not RF_MAX_FEATURES in adjusted_kwargs:
      adjusted_kwargs[RF_MAX_FEATURES] = len(self.features)
    self.random_forest = RandomForestClassifier(**adjusted_kwargs)
    self.random_forest.fit(self.df_X, self.ser_y)
    super().__init__(
        list(self.random_forest.estimators_),
        df_X.columns.tolist(), ser_y.unique().tolist())

  def _initMakeDF(self):
    """
    Sets initial values for dataframe construction.
    :return pd.DataFrame, RandomForestClassifier, int (length):
    """
    length = len(self.features)
    df_values = pd.DataFrame()
    df_values[-1] = np.repeat(0, length)
    df_values.index = self.features
    return df_values, copy.deepcopy(self.random_forest), length

  def makeRankDF(self, **kwargs):
    """
    Constructs a dataframe of feature ranks for importance.
    A more important feature has a lower rank (closer to 1)
    :return pd.DataFrame: columns are cn.MEAN, cn.STD, cn.STERR
    Ignore unused keyword arguments
    """
    # Construct the values
    df_values, clf, length = self._initMakeDF()
    # Construct the values dataframe
    for idx in range(self.iterations):
      clf.fit(self.df_X, self.ser_y)
      tuples = sorted(zip(clf.feature_importances_, self.features),
          key=lambda v: v[0], reverse=True)
      ser = pd.Series([1+x for x in range(length)])
      ser.index = [t[1] for t in tuples]
      df_values[idx] = ser
    del df_values[-1]
    # Prepare the result
    df_result = self._makeFeatureDF(df_values)
    return df_result.sort_values(cn.MEAN, ascending=True)

  def makeImportanceDF(self):
    """
    Constructs a dataframe of feature importances.
    :return pd.DataFrame: columns are cn.MEAN, cn.STD, cn.STERR
        index is features; sorted descending by cn.MEAN
    Ignore unused keyword arguments
    """
    # Construct the values
    df_values, clf, length = self._initMakeDF()
    # Construct the values
    for idx in range(self.iterations):
      clf.fit(self.df_X, self.ser_y)
      ser = pd.Series(clf.feature_importances_,
          index=self.features)
      df_values[idx] = ser
    del df_values[-1]
    # Prepare the result
    df_result = self._makeFeatureDF(df_values)
    return df_result.sort_values(cn.MEAN, ascending=False)

  def predict(self, df_X):
    return self.random_forest.predict(df_X)
