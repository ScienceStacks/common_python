'''Constructs features multi-class classifiers.'''

"""
MultiFeatureManager handles feature selection for
classifiers with multiple features by using multiple
BinaryFeatureManagers.
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
    ########### PUBLIC ##########
    self.binary_dct = {}

  def _makeBinaryClass(self, cls):
    """
    Constructs binary features from multiclass data.
    :param object cls:
    :return pd.Series: ser_y
      indices: same as original, in same sort order
      values: PCLASS if == cls; else NCLASS
    """
    ser = pd.Series([
        cn.PCLASS if v == cls else cn.NCLASS
        for v in self._ser_y],
        index=self._ser_y.index)
    return ser

  def run(self):
    """
    Construct the features, handling restarts by saving
    state and checkpointing.
    """
    for cls in self._ser_y.unique():
      if not cls in self.binary_dct:
        ser_y = self._makeBinaryClass(cls)
        selector = self.feature_selector_cls(self._df_X,
            ser_y, **self._sel_kwargs)
        self.binary_dct[cls] =  \
            BinaryFeatureSelector(self._df_X, ser_y,
                checkpoint_cb=checkpoint,
                **self._bfm_kwargs)
      if self.binary_dct[cls].is_done:
        continue
      self.binary_dct[cls].run()

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
  def Process(cls, is_restart=False, **kwargs):
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
