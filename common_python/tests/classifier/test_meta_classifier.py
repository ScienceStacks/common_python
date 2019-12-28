from common_python.classifier.hypergrid_harness  \
    import HypergridHarness, TrinaryClassification
from common_python.classifier import meta_classifier
from common_python.testing import helpers
import common_python.constants as cn
from common_python.tests.classifier import helpers as test_helpers

import pandas as pd
import numpy as np
from sklearn import svm
import unittest

IGNORE_TEST = False
SIZE = 3

class TestMetaClassifier(unittest.TestCase):

  def setUp(self):
    self.harness = HypergridHarness()
    self.df = self.harness.trinary.df_feature
    self.ser = self.harness.trinary.ser_label
    self.dfs = self.makeFeatureDFS(0, SIZE)
    self.mclf = meta_classifier.MetaClassifierDefault()

  def makeFeatureDFS(self, sigma, size):
    trinary = self.harness.trinary
    trinarys = self.harness.trinary.perturb(sigma, repl_int=size)
    return [t.df_feature for t in trinarys]

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertFalse(self.mclf._is_fit)

  def testFit(self):
    if IGNORE_TEST:
      return
    self.mclf.fit(self.dfs, self.ser)
    self.assertEqual(len(self.mclf.clf.coef_[0]),
        self.harness.dim_int)

  def testPredict(self):
    if IGNORE_TEST:
      return
    self.mclf.fit(self.dfs, self.ser)
    ser = self.mclf.predict(self.df)
    self.assertTrue(ser.equals(self.ser))

  def testCrossValidate(self):
    if IGNORE_TEST:
      return
    SIGMAS = [0.2, 0.5]
    accuracys = []
    for sigma in SIGMAS:
      dfs = self.makeFeatureDFS(sigma=sigma, size=SIZE)
      cv_result = self.mclf.crossValidate(dfs, self.ser, cv=10)
      accuracys.append(cv_result[0])
    self.assertGreater(accuracys[0], accuracys[1])

  def testScore(self):
    if IGNORE_TEST:
      return
    self.mclf.fit(self.dfs, self.ser)
    score_abs, score_rel = self.mclf.score(self.df, self.ser)
    self.assertEqual(score_abs, 1.0)
    self.assertEqual(score_rel, 1.0)

class TestMetaClassifierDefault(unittest.TestCase):

  def setUp(self):
    self.harness = HypergridHarness()
    self.df = self.harness.trinary.df_feature
    self.ser = self.harness.trinary.ser_label
    self.mclf = meta_classifier.MetaClassifierDefault()

  def testMakeTrainingData(self):
    if IGNORE_TEST:
      return
    dfs = [self.df for _ in range(SIZE)]
    df, ser = self.mclf._makeTrainingData(dfs, self.ser)
    self.assertTrue(df.equals(self.df))
    self.assertTrue(ser.equals(self.ser))


class TestMetaClassifierAverage(unittest.TestCase):

  def setUp(self):
    self.harness = HypergridHarness()
    self.df = self.harness.trinary.df_feature
    self.ser = self.harness.trinary.ser_label
    self.dfs = self.makeFeatureDFS(0, SIZE)
    self.mclf = meta_classifier.MetaClassifierAverage()

  def testMakeTrainingData(self):
    pass


class TestMetaClassifierAugment(unittest.TestCase):

  def setUp(self):
    self.harness = HypergridHarness()
    self.df = self.harness.trinary.df_feature
    self.ser = self.harness.trinary.ser_label
    self.dfs = self.makeFeatureDFS(0, SIZE)
    self.mclf = meta_classifier.MetaClassifierAugment()

  def testMakeTrainingData(self):
    pass
    


if __name__ == '__main__':
  unittest.main()
