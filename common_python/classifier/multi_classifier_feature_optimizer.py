'''Optimizes features for multi-class classifiers.'''

"""
MultiFeatureManager optimizes the feature for
multi-class classifiers. This is implemented by
using multiple BinaryFeatureManagers.
"""

import common_python.constants as cn
from common_python.util.persister import Persister
from common.trinary_data import TrinaryData
from common_python.classifier import util_classifier
from common_python.classifier import feature_collection
from common_python.classifier import  \
    binary_classifier_feature_optimizer as bcfo

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


class MultiClassifierFeatureOptimizer(object):
  """
  Does feature selection for binary classes.
  """

  def __init__(self,
      feature_collection_cl=\
      feature_collection.FeatureCollection,
      base_clf=svm.LinearSVC(), collection_kwargs={},
      bcfo_kwargs={}):
    """
    :param Classifier base_clf:  base classifier
        Exposes: fit, score, predict
    :param type-FeatureCollection feature_collection_cl:
    :param dict sel_kwargs: FeatureCollection parameters
    :param dict bcfo_kwargs: BinaryFeatureManager params
    """
    ########### PRIVATE ##########
    self._clf = copy.deepcopy(base_clf)
    self._feature_collection_cl = feature_collection_cl
    self._collection_kwargs = collection_kwargs
    self._bcfo_kwargs = bcfo_kwargs
    self._persister = Persister(PERSISTER_PATH)
    self._result_dct = {cn.FEATURE: [], cn.CLASS: [],
        cn.SCORE: []}
    self._binary_dct = {}  # binary classifier optimizers
    ########### PUBLIC ##########
    self.feature_dct = {}  # key: class; value: features
    self.score_dct = {}
    self.best_score_dct = {}

  def checkpoint(self):
    self._persister.set(self)

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
    for cl in ser_y.unique():
      if not cl in self._binary_dct:
        self._binary_dct[cl] =  \
            bcfo.BinaryClassifierFeatureOptimizer(
                checkpoint_cb=self.checkpoint,
                **self._bcfo_kwargs)
      if self._binary_dct[cl].is_done:
        continue
      else:
        ser_y_cl = util_classifier.makeOneStateSer(
            ser_y, cl)
        collection = self._feature_collection_cl(df_X,
            ser_y_cl, **self._collection_kwargs)
        self._binary_dct[cl].fit(df_X, ser_y_cl)
        self.feature_dct[cl] =  \
            self._binary_dct[cl].features
        self.score_dct[cl] =  \
            self._binary_dct[cl].score
        self.best_score_dct[cl] =  \
            self._binary_dct[cl].best_score

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
  def getPersister(cls):
    if os.path.isfile(PERSISTER_PATH):
      return Persister(PERSISTER_PATH)
    else:
      return None

  @staticmethod
  def removePersister(cls):
    if cls.getPersister() is not None:
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
      cls.removePersister()
    persister = cls.getPersister()
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
