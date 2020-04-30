'''Optimizes features for multi-class classifiers.'''

"""
MultiFeatureManager optimizes the feature for
multi-class classifiers. This is implemented by
using multiple BinaryFeatureManagers.
"""

from common_python.util.persister import Persister
from common.trinary_data import TrinaryData
from common_python.classifier import util_classifier
from common_python.classifier import feature_selector
from common_python.classifier import  \
    binary_feature_manager

import copy
import numpy as np
import os
import pandas as pd
import random
from sklearn import svm


# Default checkpoint callback
DIR = os.path.dirname(os.path.abspath(__file__))
PERSISTER_PATH = os.path.join(DIR,
    "multi_feature_manager.pcl")


class MultiFeatureManager(object):
  """
  Does feature selection for binary classes.
  Exposes the following instance variables
    1. binary_dct - dict
         key: class, value binary_feature_manager
    2. MultiFeatureManager.get()
         return: instance from serialization
    3. MultiFeatureManager.run()
         Acquires data and runs
  """

  def __init__(self, df_X, ser_y,
      feature_selector_cls=FeatureSelector
      base_clf=svm.LinearSVC(), sel_kwargs,
      **bfm_kwargs)
      ):
    """
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: binary class values (0, 1)
    :param Classifier base_clf:  base classifier
        Exposes: fit, score, predict
    :param type-FeatureSelector feature_selector_cls:
    :param dict sel_kwargs: FeatureSelector parameters
    :param dict bfm_kwargs: BinaryFeatureManager params
    """
    ########### PRIVATE ##########
    self._clf = base_clf
    self._df_X = df_X
    self._ser_y = ser_y
    self._selector_cls = feature_selector_cls
    self._sel_kwargs = sel_kwargs
    self._bfm_kwargs = bfm_kwargs
    self._persister = Persister(PERSISTER_PATH)
    self._result_dct = {cn.FEATURE: [], cn.CLASS: [],
        cn.SCORE: []}
    ########### PUBLIC ##########
    self.binary_dct = {}

  def fit(self, df_X, ser_y):
    """
    Construct the features, handling restarts by saving
    state and checkpointing.
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: binary class values (0, 1)
    """
    for cls in ser_y.unique():
      if not cls in self.binary_dct:
        ser_y_cls = self.util_classifier.makeOneStateSer(
            ser_y, cls)
        selector = self.feature_selector_cls(df_X,
            ser_y_cls, **self._sel_kwargs)
        self.binary_dct[cls] =  \
            BinaryFeatureSelector(df_X, ser_y_cls
                checkpoint_cb=checkpoint,
                **self._bfm_kwargs)
      if self.binary_dct[cls].is_done:
        continue
      self.binary_dct[cls].run()

  # FIXME: Is organized for persister
  @classmethod
  def crossValidate(cls, df_X, ser_y, 
      holdouts=1, num_iter=10, **kwargs):
    """
    Constructs cross validated features.
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: binary class values (0, 1)
    :param int holdouts:
    :param int num_iter: number of cross validates
    :param dict kwargs: passed to 
        MultiClassifierFeatureOptimizer constructor
    :return pd.DataFrame:
        columns: FEATURE, CLASS, SCORE
        SCORE is in [0, 1], fraction of
          cross validations in which
          the feature appears for the class.
    """
    classes = ser_y.unique()
    for _ in range(num_iter):
      # Find features for this iteration using train data
      optimizer = cls(**kwargs)
      train_idxs, test_idxs =  \
          util_classier.partitionByState(ser_y,
          holdouts=holdouts)
      optimizer.fit(df_X.loc[train_idxs, :],
          ser_y.loc[train_idxs])
      # Record the features created
      for cl in classes:
        ser_y_cls = util_classifier.makeOneStateSer(
            ser_y, cl)
        clf = copy.deepcopy(self._base_clf)
        score = util_classifier.scoreFeatures(
            clf, df_X, ser_y_cls,
            features=optimizer.feature_dct[cl], 
            train_idxs=train_idxs, test_idxs=test_idxs)
        for feature in optimizer.features:
          self.result_dct[cn.FEATURE] = feature
          self.result_dct[cn.CLASS] = cl
          self.result_dct[cn.SCORE] = score
    # Construct the dataframe
    df = pd.DataFrame(self.result_dct)
    df_result = pd.DataFrame(df.groupby(
        [cn.FEATURE, cn.CLASS]).count())
    df_result = df_result.applymap(lambda v: v/num_iter)
    df_result = df_result.reset_index()
    return df_result

  @staticmethod
  def _getPersister(cls)
    if os.path.isfile(PERSISTER_PATH):
      return Persister(PERSISTER_PATH)
    else:
      return None

  @staticmethod
  def _removePersister(cls)
    if cls._getPersister() is not None:
      os.remove(PERSISTER_PATH)

  @classmethod
  def process(cls, is_restart=False, **kwargs):
    """
    Acquires data and manages persister file.
    :param bool is_restart: delete any existing persister
    :param dict kwargs: parameters for MultiFeatureManager
    ;return MultiFeatureManager:
    """
    if is_restart:
      cls._removePersister()
    persister = cls._getPersister()
    if persister is not None:
      manager = persister.get()
    else:
      trinary = trinary_data.TrinaryData(
          is_averaged=False, is_dropT1=False)
      manager = MultiFeatureManager(trinary.df_X, trinary.ser_y,
          **kwargs)
    manager.run()
    return manager

  @classmethod
  def get(cls):
    """
    Provides the MultiFeatureManager in a persister file.
    """
    persister = Persister(PERSISTER_PATH)
    return persister.get()
