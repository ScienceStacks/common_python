import common_python.constants as cn
from common_python.util.persister import Persister
from common_python.testing import helpers
from common_python.classifier  \
    import binary_classifier_feature_optimizer as bcfo
from common_python.tests.classifier import helpers as test_helpers
from common_python.classifier import feature_collection
from common_python.testing import helpers
from common_python.util.persister import Persister

import copy
import os
import pandas as pd
import numpy as np
from sklearn import svm
import unittest

IGNORE_TEST = False

CLASS = 1
DF_X, SER_Y = test_helpers.getDataLong()
SER_Y = pd.Series([
    cn.PCLASS if v == CLASS else cn.NCLASS
    for v in SER_Y], index=SER_Y.index)


class TestBinaryClassifierFeatureOptimizer(
    unittest.TestCase):
  
  def _init(self):
    self.df_X = copy.deepcopy(DF_X)
    self.ser_y = copy.deepcopy(SER_Y)
    self.optimizer =  \
        bcfo.BinaryClassifierFeatureOptimizer()

  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertTrue(isinstance(self.optimizer._base_clf,
        svm.LinearSVC))

  def testUpdateIteration(self):
    if IGNORE_TEST:
      return
    self.optimizer._updateIteration()
    self.assertEqual(self.optimizer._iteration, 0)

  def testEvaluate(self):
    if IGNORE_TEST:
      return
    score = self.optimizer._evaluate(self.df_X,
        self.ser_y, features=self.df_X.columns.tolist())
    self.assertGreater(score, 0.80)
    score2 = self.optimizer._evaluate(self.df_X,
        self.ser_y,
        features=self.df_X.columns.tolist()[0:1])
    self.assertGreater(score, score2)

  def testFit(self):
    if IGNORE_TEST:
      return
    self._init()
    def test(max_iter, desired_accuracy=1.0):
      optimizer = bcfo.BinaryClassifierFeatureOptimizer(
          max_iter=max_iter, desired_accuracy=desired_accuracy)
      optimizer.fit(self.df_X, self.ser_y)
      self.assertTrue(isinstance(optimizer.score, float))
      self.assertTrue(isinstance(optimizer.all_score,
          float))
      self.assertGreater(len(optimizer.selecteds), 0)
      return optimizer
    #
    opt1 = test(1)
    opt50 = test(50)
    self.assertGreater(opt50.score, opt1.score)
    opt50a = test(50, desired_accuracy=0.5)
    diff = set(opt50a.selecteds).symmetric_difference(
        opt1.selecteds)
    self.assertEqual(len(diff), 0)


if __name__ == '__main__':
  unittest.main()
