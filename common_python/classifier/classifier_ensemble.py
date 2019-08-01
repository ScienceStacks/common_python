"""Maniuplations of an ensemble of classifiers for same data."""

import common_python.constants as cn
from common_python.util import util

import collections
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


CrossValidationResult = collections.namedtuple(
    "CrossValidationResult", "mean std ensemble")


class ClassifierEnsemble(object):

  def __init__(self, classifiers, features, classes):
    """
    :param list-Classifier classifiers: classifiers
    """
    self.classifiers = classifiers
    self.features = features
    self.classes = classes

  @classmethod
  def _crossValidate(cls, classifier, df_X, ser_y,
      iterations=5, holdouts=1):
    """
    Does cross validation wth holdouts for each state.
    :param Classifier classifier: untrained classifier with fit, score methods
    :param pd.DataFrame df_X: columns of features, rows of instances
    :param pd.Series ser_y: state values
    :param int interations: number of cross validations done
    :param int holdouts: number of instances per state in test data
    :return CrossValidationResult:
    Notes
      1. df_X, ser_y must have the same index
    """
    def sortIndex(container, indices):
      container = container.copy()
      container.index = indices
      return container.sort_index()
    def partitionData(container, all_indices, test_indices):
      train_indices = list(set(all_indices).difference(test_indices))
      if isinstance(container, pd.DataFrame):
        container_test = container.loc[test_indices, :]
        container_train = container.loc[train_indices, :]
      else:
        container_test = container.loc[test_indices]
        container_train = container.loc[train_indices]
      return container_train, container_test
    #
    scores = []
    classifiers = []
    classes = ser_y.unique()
    indices = ser_y.index.tolist()
    for _ in range(iterations):
      # Construct test set
      new_classifier = copy.deepcopy(classifier)
      classifiers.append(new_classifier)
      indices = np.random.permutation(indices)
      df_X = sortIndex(df_X, indices)
      ser_y = sortIndex(ser_y, indices)
      test_indices = []
      for cl in classes:
        ser = ser_y[ser_y == cl]
        if len(ser) <= holdouts:
          raise ValueError("Class %s has fewer than %d holdouts" %
              (cl, holdouts))
        idx = ser.index[0:holdouts].tolist()
        test_indices.extend(idx)
      df_X_train, df_X_test = partitionData(df_X, indices, test_indices)
      ser_y_train, ser_y_test = partitionData(ser_y, indices, test_indices)
      new_classifier.fit(df_X_train, ser_y_train)
      score = new_classifier.score(df_X_test, ser_y_test)
      scores.append(score)
    return CrossValidationResult(
        mean=np.mean(scores), 
        std=np.std(scores), 
        ensemble=cls(classifiers, df_X.columns.tolist(), ser_y.unique().tolist())
        )

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
      class_selection=None, is_plot=True, **kwargs):
    """
    Plots the rank of features for the top valued features.
    :param int top:
    :param bool is_plot: produce the plot
    :param ax, fig: matplotlib
    :param dict kwargs: keyword arguments for plot
    """
    # Data preparation
    df = self.makeRankDF(class_selection=class_selection)
    self._plot(df, "Rank", top, fig, ax, is_plot, **kwargs)

  def plotImportance(self, top=None, fig=None, ax=None, 
      is_plot=True, class_selection=None, **kwargs):
    """
    Plots the rank of features for the top valued features.
    :param int top:
    :param bool is_plot: produce the plot
    :param ax, fig: matplotlib
    :param dict kwargs: keyword arguments for plot
    """
    df = self.makeImportanceDF(class_selection=class_selection)
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
    df_result[cn.STERR] = df_result[cn.STD] / np.sqrt(len(self.classifiers))
    return df_result
    
   
##################################################################### 
class LinearSVMEnsemble(ClassifierEnsemble):

  def _orderFeatures(self, clf, class_selection):
    """
    Orders features by descending value of importance.
    :param int class_selection: restrict analysis to a single class
    :return list-int:
    """
    if class_selection is None:
      coefs = [max([np.abs(x) for x in xv]) for xv in zip(*clf.coef_)]
    else:
      coefs = [np.abs(x) for x in clf.coef_[class_selection]]
    length = len(coefs)
    sorted_tuples = np.argsort(coefs).tolist()
    # Calculate rank in descending order
    result = [length - sorted_tuples.index(v) for v in range(length)]
    return result

  def makeRankDF(self, class_selection=None):
    """
    Constructs a dataframe of feature ranks for importance.
    A more important feature has a lower rank (closer to 0)
    :param int class_selection: restrict analysis to a single class
    :return pd.DataFrame: columns are cn.MEAN, cn.STD, cn.STERR
    """
    df_values = pd.DataFrame()
    for idx, clf in enumerate(self.classifiers):
      df_values[idx] = pd.Series(self._orderFeatures(clf, class_selection),
          index=self.features)
    df_result = self._makeFeatureDF(df_values)
    return df_result.sort_values(cn.MEAN)

  def makeImportanceDF(self, class_selection=None):
    """
    Constructs a dataframe of feature importances.
    :param int class_selection: restrict analysis to a single class
    :return pd.DataFrame: columns are cn.MEAN, cn.STD, cn.STERR
    The importance of a feature is the maximum absolute value of its coefficient
    in the collection of classifiers for the multi-class vector (one vs. rest, ovr).
    """
    ABS_MEAN = "abs_mean"
    df_values = pd.DataFrame()
    for idx, clf in enumerate(self.classifiers):
      if class_selection is None:
        values = [max(np.abs(x) for x in xv) for xv in  zip(*clf.coef_)]
      else:
        values = [x for x in clf.coef_[class_selection]]
      df_values[idx] = pd.Series(values, index=self.features)
    df_result = self._makeFeatureDF(df_values)
    df_result[ABS_MEAN] = [np.abs(x) for x in df_result[cn.MEAN]]
    df_result = df_result.sort_values(ABS_MEAN, ascending=False)
    del df_result[ABS_MEAN]
    return df_result

  @classmethod
  def crossValidate(cls, df_X, ser_y, classifier_args=None, **kwargs):
    """
    Cross validation for SVM.
    :param pd.DataFrame df_X: feature vectors
    :param pd.Series ser_y: classes
    :param dict classifier_args: arguments to use in classifier
        constructor
    :param dict kwargs: arguments passed to crossValidate
    """
    if classifier_args is None:
      classifier_args = {}
    clf = svm.LinearSVC(**classifier_args)
    return cls._crossValidate(clf, df_X, ser_y, **kwargs)
    
  
##################################################################### 
class RandomForestEnsemble(ClassifierEnsemble):

  def __init__(self, df_X, ser_y, num_iterations=5, **kwargs):
    """
    :param pd.DataFrame df_X:
    :param pd.Series ser_y:
    :param int num_iterations: Number of random forests created
        to obtain a distribution
    :param dict kwargs: arguments passed to classifier
    """
    self.df_X = df_X
    self.ser_y = ser_y
    self.num_iterations = num_iterations
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
    super().__init__(list(self.random_forest.estimators_),
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
    for idx in range(self.num_iterations):
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

  def makeImportanceDF(self, **kwargs):
    """
    Constructs a dataframe of feature importances.
    :return pd.DataFrame: columns are cn.MEAN, cn.STD, cn.STERR
        index is features; sorted descending by cn.MEAN
    Ignore unused keyword arguments
    """
    # Construct the values
    df_values, clf, length = self._initMakeDF()
    # Construct the values
    for idx in range(self.num_iterations):
      clf.fit(self.df_X, self.ser_y)
      ser = pd.Series(clf.feature_importances_,
          index=self.features)
      df_values[idx] = ser
    del df_values[-1]
    # Prepare the result
    df_result = self._makeFeatureDF(df_values)
    return df_result.sort_values(cn.MEAN, ascending=False)
