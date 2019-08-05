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
  """
  A classifier formed by an ensemble of Linear SVM classifiers. Does
  feature selection based on the rank of features.
  """
 
  def __init__(self, df_X, ser_y, filter_high_rank=None,
      holdouts=1, size=1, **kwargs):
    self.clf = svm.LinearSVC(**kwargs)
    self.filter_high_rank = filter_high_rank
    self.holdouts = holdouts
    self.size = size
    super().__init__(None, None, None, None)

  def fit(self, df_X, ser_y):
    """
    :param pd.DataFrame df_X: feature vectors
    :param pd.Series ser_y: classes
    :param int filter_high_rank: mean ranks beyond which features are dropped
    :param int size: number of classifiers to create
    :param dict kwargs: arguments to use in classifier
    """
    result = classifier_collection.ClassifierCollection(
        self.clf, df_X, ser_y, count, holdouts=holdouts)
    if filter_high_rank is None:
      return result
    # Select the features
    df_rank = result.ensemble.makeRankDF()
    df_rank = df_rank.loc[df_rank.index[0:filter_high_rank], :]
    columns = df_rank.index.tolist()
    df_X_filtered = df_X[columns]
    collection = classifier_collection.ClassifierCollection(
        clf, df_X_filered, ser_y, count, holdouts=holdouts)
    super().__init__(collection.clfs, collection.features,
        collection.classes, collection.scores)

  def predict(self, df_X):
    pass

  def score(self, df_X, ser_y):
    pass

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
  def crossValidate(cls, df_X, ser_y, filter_high_rank=None,
      classifier_args=None, **kwargs):
    """
    Cross validation for SVM.
    :param pd.DataFrame df_X: feature vectors
    :param pd.Series ser_y: classes
    :param int filter_high_rank: mean ranks beyond which features are dropped
    :param dict classifier_args: arguments to use in classifier
        constructor
    :param dict kwargs: arguments passed to crossValidate
    """
    if classifier_args is None:
      classifier_args = {}
    clf = svm.LinearSVC(**classifier_args)
    result = cls._crossValidate(clf, df_X, ser_y, **kwargs)
    if filter_high_rank is None:
      return result
    # Select the features
    df_rank = result.ensemble.makeRankDF()
    df_rank = df_rank.loc[df_rank.index[0:filter_high_rank], :]
    columns = df_rank.index.tolist()
    df_X_filtered = df_X[columns]
    return cls._crossValidate(clf, df_X_filtered, ser_y, **kwargs)
    
  
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
