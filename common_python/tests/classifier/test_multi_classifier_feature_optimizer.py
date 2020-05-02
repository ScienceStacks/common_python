import common_python.constants as cn
from common_python.util.persister import Persister
from common_python.testing import helpers
from common_python.classifier  \
    import multi_classifier_feature_optimizer as mcfo
from common_python.tests.classifier import helpers as test_helpers
from common_python.classifier import feature_collection
from common_python.testing import helpers

import copy
import os
import pandas as pd
import numpy as np
from sklearn import svm
import unittest

IGNORE_TEST = False

DF_X, SER_Y = test_helpers.getDataLong()


class TestMultiClassifierFeatureOptimizer(
    unittest.TestCase):
  
  def _init(self):
    self.df_X = copy.deepcopy(DF_X)
    self.ser_y = copy.deepcopy(SER_Y)
    self.optimizer =  \
        mcfo.MultiClassifierFeatureOptimizer(
        num_exclude_iter=1,
        bcfo_kwargs=dict(max_iter=100))

  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertTrue(isinstance(self.optimizer._base_clf,
        svm.LinearSVC))

  def testFit(self):
    if IGNORE_TEST:
      return
    self.optimizer.fit(self.df_X, self.ser_y)
    for cl in self.optimizer.feature_dct.keys():
      self.assertGreaterEqual(
          len(self.optimizer.feature_dct[cl]), 1)



if __name__ == '__main__':
  unittest.main()
