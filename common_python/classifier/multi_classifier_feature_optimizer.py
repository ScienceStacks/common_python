'''Optimizes features for multi-class classifiers.'''

"""
MultiClassifierFeatureOPtimizer selects features so
as to optimize the accuracy of MultiClassifier.
This is implemented by using multiple
BinaryFeatureManagers as uses checkpoints (via
Persister) because it is long running.

The hyperparameters are:
  num_exclude_iter: number of iterations in which results
     from previous iterations are excluded. These are
     referred to as exclude iterations.
"""

import common_python.constants as cn
from common.trinary_data import TrinaryData
from common_python.classifier import util_classifier
from common_python.classifier import feature_collection
from common_python.classifier import  \
    binary_classifier_feature_optimizer as bcfo
from common_python.util.persister import Persister

import collections
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
NUM_EXCLUDE_ITER = 5  # Number of exclude iterations


FitResult = collections.namedtuple("FitResult",
    "iteration selects selects_score all_score excludes")
#    iteration: of excludes
#    selects: list of features selected
#    selects_score: score for classifier with selects
#    all_score: score for classifier with all non-excludes
#    excludes: list of features excluded


class MultiClassifierFeatureOptimizer(object):
  """
  Does feature selection for binary classes.
  """

  def __init__(self,
      feature_collection_cl=\
      feature_collection.FeatureCollection,
      base_clf=svm.LinearSVC(), is_restart=True,
      persister=None,
      num_exclude_iter=NUM_EXCLUDE_ITER,
      collection_kwargs={},
      bcfo_kwargs={}):
    """
    :param Classifier base_clf:  base classifier
        Exposes: fit, score, predict
    :param type-FeatureCollection feature_collection_cl:
    :param dict collection_kwargs:
        FeatureCollection parameters
    :param Persister persister:
    :param dict bcfo_kwargs:
         BinaryClassifierFeatureOptimizer parameters
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
      self._num_exclude_iter = num_exclude_iter
      ########### PUBLIC ##########
      self.fit_result_dct = {}  # list of FitResult
      self.feature_dct = {}  # key: class; value: features
      self.score_dct = {}
      self.all_score_dct = {}

  def checkpoint(self):
    self._persister.set(self)

  # FIXME: (1) Accumulating results by excludes
  #        (2) Eliminate dcts in this code
  #        (3) Fix tests
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
      self.fit_result_dct[cl] = []
      for count_exclude_iter in range(
          self._num_exclude_iter):
        excludes = []
        [excludes.extend(f.selects) 
            for f in self.fit_result_dct[cl]]
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
          sel_features =  \
              list(set(df_X.columns).difference(excludes))
          self._binary_dct[cl].fit(
              df_X[sel_features], ser_y_cl)
          self.feature_dct[cl] =  \
              self._binary_dct[cl].selects
          self.score_dct[cl] =  \
              self._binary_dct[cl].score
          self.all_score_dct[cl] =  \
              self._binary_dct[cl].all_score
      fit_result = FitResult(
          iteration=count_exclude_iter,
          selects=self._binary_dct[cl].selects,
          selects_score = self._binary_dct[cl].score,
          all_score=self._binary_dct[cl].all_score,
          excludes=excludes)
      self.fit_result_dct[cl].append(fit_result)
    self.checkpoint()
