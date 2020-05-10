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
    self._init()
    self.optimizer.fit(self.df_X, self.ser_y)
    for cl in self.optimizer.fit_result_dct.keys():
      self.assertEqual(
          len(self.optimizer.fit_result_dct[cl]), 1)
      fit_results = self.optimizer.fit_result_dct[cl]
      self.assertGreaterEqual(
          len(fit_results[0].sels), 1)

  def testFit2(self):
    if IGNORE_TEST:
      return
    self._init()
    CLS = 0
    NUM_EXCLUDE_ITER = 3
    optimizer = mcfo.MultiClassifierFeatureOptimizer(
        num_exclude_iter=NUM_EXCLUDE_ITER,
        bcfo_kwargs=dict(max_iter=2))
    optimizer.fit(self.df_X, self.ser_y)
    fit_results = optimizer.fit_result_dct[CLS]
    self.assertEqual(len(fit_results), NUM_EXCLUDE_ITER)
    features0 = fit_results[0].sels
    features1 = fit_results[1].sels
    sames = set(features0).intersection(features1)
    self.assertEqual(len(sames), 0)



if __name__ == '__main__':
  unittest.main()
