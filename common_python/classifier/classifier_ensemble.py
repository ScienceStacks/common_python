"""
Ensemble of classifiers formed by training with different data.
The underlying classifier is called the base classifier.
1. Requirements
  a. The base classifier must expose methods for fit, predict, score.
  b. Sub-class ClassifierDescriptor and implement getImportance
2. The EnsembleClassifer is trained using randomly selecting data
subsets using a specified number of holdouts.
3. The ClassifierEnsemble exposes fit, predict, score.
4. In addition, features can be exampled and plotted
  a. Importance is float for a feature that indicates its contribution
     to classifications
  b. Rank is an ordering of features based on their importance
"""

import common_python.constants as cn
from common_python.util import util
from common_python.classifier.classifier_collection  \
    import ClassifierCollection

import collections
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm


###############################################
class ClassifierDescriptor(object):
  # Describes a classifier used to create an ensemble

  def __init__(self):
    self.clf = None # Must assign to a classifier Type
    raise RuntimeError("Must override ClassifierDescriptor.__init__")

  def getImportance(self, clf, **kwargs):
    """
    Returns the importances of features.
    :return list-float:
    """
    raise RuntimeError(
        "Must override ClassifierDescriptor.getImportance")


class ClassifierDescriptorSVM(ClassifierDescriptor):
  # Descriptor information needed for SVM classifiers
  # Descriptor is for one-vs-rest. So, there is a separate
  # classifier for each class.
  
  def __init__(self, clf=svm.LinearSVC()):
    self.clf = clf

  def getImportance(self, clf, class_selection=None):
    """
    Calculates the importances of features.
    :param Classifier clf:
    :param int class_selection: class for which importance is computed
    :return list-float:
    """
    if class_selection is None:
      # If none specified, choose the largest value.
      coefs = [max([np.abs(x) for x in xv]) for xv in zip(*clf.coef_)]
    else:
      coefs = [np.abs(x) for x in clf.coef_[class_selection]]
    return coefs


###################################################
class ClassifierEnsemble(ClassifierCollection):
 
  def __init__(self, clf_desc=ClassifierDescriptorSVM(),
      holdouts=1, size=1,
      filter_high_rank=None,
      **kwargs):
    """
    :param ClassifierDescriptor clf_desc:
    :param int holdouts: number of holdouts when fitting
    :param int size: size of the ensemble
    :param int/None filter_high_rank: maxium rank considered
    :param dict kwargs: keyword arguments used by parent classes
    """
    self.clf_desc = clf_desc
    self.filter_high_rank = filter_high_rank
    self.holdouts = holdouts
    self.size = size
    super().__init__(**kwargs)

  def fit(self, df_X, ser_y):
    """
    Fits the number of classifiers desired to the data with
    holdouts. Current selects holdouts independent of class.
    :param pd.DataFrame df_X: feature vectors; indexed by instance
    :param pd.Series ser_y: classes; indexed by instance
    """
    collection = ClassifierCollection.makeByRandomHoldout(
        self.clf_desc.clf, df_X, ser_y, self.size, 
        holdouts=self.holdouts)
    self.update(collection)
    if self.filter_high_rank is None:
      return
    # Select the features
    df_rank = self.makeRankDF()
    df_rank_sub = df_rank.loc[
        df_rank.index[0:self.filter_high_rank], :]
    columns = df_rank_sub.index.tolist()
    df_X_sub = df_X[columns]
    collection = ClassifierCollection.makeByRandomHoldout(
        self.clf_desc.clf, df_X_sub, ser_y, self.size, 
        holdouts=self.holdouts)
    self.update(collection)

  def predict(self, df_X):
    """
    Default prediction algorithm. Reports probability of each class.
    The probability of a class is the fraction of classifiers that
    predict that class.
    :param pd.DataFrame: features, indexed by instance.
    :return pd.DataFrame:  probability by class;
        columns are classes; index by instance
    Assumes that self.clfs has been previously populated with fitted
    classifiers.
    """
    # Change to an array of array of features
    DUMMY_COLUMN = "dummy_column"
    if isinstance(df_X, pd.Series):
      df_X = pd.DataFrame(df_X)
      df_X = df_X.T
    array = df_X.values
    array = array.reshape(len(df_X.index), len(df_X.columns)) 
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

  def score(self, df_X, ser_y):
    """
    Evaluates the accuracy of the ensemble classifier for
    the instances pvodied.
    :param pd.DataFrame df_X: columns are features, rows are instances
    :param pd.Series ser_y: rows are instances, values are classes
    Returns a float in [0, 1] which is the accuracy of the
    ensemble classifier.
    Assumes that self.clfs has been previously populated with fitted
    classifiers.
    """
    df_predict = self.predict(df_X)
    accuracies = []
    for instance in ser_y.index:
      # Accuracy is the probability of selecting the correct class
      accuracy = df_predict.loc[instance, ser_y.loc[instance]]
      accuracies.append(accuracy)
    return np.mean(accuracies)

  def _orderFeatures(self, clf, class_selection=None):
    """
    Orders features by descending value of importance.
    :param int class_selection: restrict analysis to a single class
    :return list-int:
    """
    coefs = self.clf_desc.getImportance(clf,
        class_selection=class_selection)
    length = len(coefs)
    sorted_tuples = np.argsort(coefs).tolist()
    # Calculate rank in descending order
    result = [length - sorted_tuples.index(v) for v in range(length)]
    return result

  def makeRankDF(self, class_selection=None):
    """
    Constructs a dataframe of feature ranks for importance,
    where the rank is the feature order of importance.
    A more important feature has a lower rank (closer to 0)
    :param int class_selection: restrict analysis to a single class
    :return pd.DataFrame: columns are cn.MEAN, cn.STD, cn.STERR
    """
    df_values = pd.DataFrame()
    for idx, clf in enumerate(self.clfs):
      df_values[idx] = pd.Series(self._orderFeatures(clf,
          class_selection=class_selection),
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
    for idx, clf in enumerate(self.clfs):
      values = self.clf_desc.getImportance(clf,
         class_selection=class_selection)
      df_values[idx] = pd.Series(values, index=self.features)
    df_result = self._makeFeatureDF(df_values)
    df_result[ABS_MEAN] = [np.abs(x) for x in df_result[cn.MEAN]]
    df_result = df_result.sort_values(ABS_MEAN, ascending=False)
    del df_result[ABS_MEAN]
    return df_result

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
