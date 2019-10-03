"""
Encapsulate Random forest as a ClassifierEnsemble.
This provides uniform handling of classifiers. Also, provides:
1. predictions are probabilistic
2. access to plotting
"""

import common_python.constants as cn
from common_python.util import util
from common_python.classifier.classifier_collection  \
    import ClassifierCollection

import collections
import copy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

RF_ESTIMATORS = "n_estimators"
RF_MAX_FEATURES = "max_features"
RF_BOOTSTRAP = "bootstrap"
RF_DEFAULTS = {
    RF_ESTIMATORS: 500,
    RF_BOOTSTRAP: True,
    }

  
class ClassifierEnsembleRandomForest(ClassifierEnsemble):

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
