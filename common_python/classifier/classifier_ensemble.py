"""Maniuplations of an ensemble of classifiers for same data."""

import common_python.constants as cn

import collections
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
  def crossVerify(cls, classifier, df_X, ser_y,
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

  def makeRankDF(self):
    """
    Constructs a dataframe of feature ranks for importance.
    A more important feature has a lower rank (closer to 0)
    :return pd.DataFrame: columns are cn.MEAN, cn.STD
    """
    df = pd.DataFrame()
    for idx, clf in enumerate(self.classifiers):
      features = self.orderFeatures(clf)
      df[idx] = pd.Series(range(len(features)), index=features)
    df_result = pd.DataFrame({
        cn.MEAN: df.mean(axis=1),
        cn.STD: df.std(axis=1) / np.sqrt(len(self.classifiers)),
        })
    df_result = df_result.sort_values(cn.MEAN)
    return df_result

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
    ax.set_xticklabels(indices, rotation=90)
    ax.set_xlabel('Gene Group')
    ax.set_ylabel('Rank')
    if cn.PLT_TITLE in kwargs:
      ax.set_title(kwargs[cn.PLT_TITLE])
    if is_plot:
      plt.show()
    return fig, ax
    
   
class LinearSVMEnsemble(ClassifierEnsemble):

  def orderFeatures(self, clf):
    """
    Orders features by descending value of importance.
    :return list-object:
    """
    coefs = [max([np.abs(x) for x in xv]) for xv in zip(*clf.coef_)]
    sorted_tuples = sorted(zip(self.features, coefs),
        key=lambda v: v[1], reverse=True)
    result = [t[0] for t in sorted_tuples]
    return result
    
    
    
