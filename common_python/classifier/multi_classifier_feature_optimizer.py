'''Optimizes features for multi-class classifiers.'''

"""
MultiFeatureManager optimizes the feature for
multi-class classifiers. This is implemented by
using multiple BinaryFeatureManagers.
"""

import common_python.constants as cn
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
      base_clf=svm.LinearSVC(), is_restart=True,
      collection_kwargs={},
      persister=None,
      bcfo_kwargs={}):
    """
    :param Classifier base_clf:  base classifier
        Exposes: fit, score, predict
    :param type-FeatureCollection feature_collection_cl:
    :param dict sel_kwargs: FeatureCollection parameters
    :param Persister persister:
    :param dict bcfo_kwargs: BinaryFeatureManager params
    """
    if persister is None:
      self._persister = Persister(PERSISTER_PATH)
    else:
      self._persister = persister
    if is_restart:
      ########### PRIVATE ##########
      self._base_clf = copy.deepcopy(base_clf)
      self._feature_collection_cl = feature_collection_cl
      self._collection_kwargs = collection_kwargs
      self._bcfo_kwargs = bcfo_kwargs
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
      ser_y_cl = util_classifier.makeOneStateSer(
          ser_y, cl)
      if not cl in self._binary_dct:
        collection = self._feature_collection_cl(df_X,
            ser_y_cl, **self._collection_kwargs)
        self._binary_dct[cl] =  \
            bcfo.BinaryClassifierFeatureOptimizer(
                base_clf=self._base_clf,
                checkpoint_cb=self.checkpoint,
                feature_collection=collection,
                **self._bcfo_kwargs)
      if self._binary_dct[cl].is_done:
        continue
      else:
        self._binary_dct[cl].fit(df_X, ser_y_cl)
        self.feature_dct[cl] =  \
            self._binary_dct[cl].features
        self.score_dct[cl] =  \
            self._binary_dct[cl].score
        self.best_score_dct[cl] =  \
            self._binary_dct[cl].best_score
    self.checkpoint()
