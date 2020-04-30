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

IGNORE_TEST = True

CLASS = 1
DF_X, SER_Y = test_helpers.getDataLong()
CLF = svm.LinearSVC()
SER_Y = pd.Series([
    cn.PCLASS if v == CLASS else cn.NCLASS
    for v in SER_Y], index=SER_Y.index)


class TestBinaryClassifierFeatureOptimizer(unittest.TestCase):
  
  def _init(self):
    self.df_X = copy.deepcopy(DF_X)
    self.ser_y = copy.deepcopy(SER_Y)
    self.optimizer =  \
        bcfo.BinaryClassifierFeatureOptimizer(
        self.df_X, self.ser_y, CLF)

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

  def testMakeTestIndices(self):
    if IGNORE_TEST:
      return
    test_idxs = self.optimizer._makeTestIndices()
    classes = self.ser_y.loc[test_idxs]
    cls_set = set(classes)
    self.assertEqual(len(cls_set), 2)
    p_classes = [c for c in classes if c == cn.PCLASS]
    n_classes = [c for c in classes if c == cn.NCLASS]
    self.assertEqual(len(p_classes), len(n_classes))

  def testRun(self):
    # TESTING
    self._init()
    optimizer = bcfo.BinaryClassifierFeatureOptimizer(
        self.df_X, self.ser_y, CLF, max_iter=2)
    optimizer.run()
    import pdb; pdb.set_trace()


if __name__ == '__main__':
  unittest.main()
