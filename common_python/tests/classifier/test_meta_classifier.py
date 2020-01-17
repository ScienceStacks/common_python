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
from sklearn.linear_model import LogisticRegression


IGNORE_TEST = False
SIZE = 3
NUM_REPL = 3
SIGMA_TRAIN = 0.1
SIGMA_TEST = 0.3
SCORE_PERFECT = 1.0
TEST_LOGISTIC = False  # Do tests for logistic classifier


def testScoreGeneric(testcase, sigma=0.2, num_repl=3):
  """
  Generic test for a MetaClassifier
  :param TestCase testcase: TestCase for class under test
  """
  # Fit to training data 
  testcase.mclf.fit(testcase.dfs_train, testcase.ser)
  # Score on training data
  result1 = testcase.mclf.score(testcase.dfs_train[0], testcase.ser)
  # Score on test data
  result2 = testcase.mclf.score(testcase.df_test, testcase.ser)
  #
  testcase.assertGreater(result1.abs, result2.abs)

def testMakeTrainingDataGeneric(testcase):
  """
  Generic test for make training data.
  Assumes instance variables in setUp for class under tests.
  :param TestCase testcase: TestCase for class under test
  """
  df, ser = testcase.mclf._makeTrainingData(
      testcase.dfs_train, testcase.ser)
  if len(ser) == len(testcase.ser):
    testcase.assertTrue(ser.equals(testcase.ser))
  testcase.assertTrue(helpers.isValidDataFrame(df,
      expected_columns=testcase.dfs_train[0].columns))

def setUpGeneric(testcase):
  testcase.harness = HypergridHarness()
  trinarys = testcase.harness.trinary.perturb(
      sigma=SIGMA_TRAIN, num_repl=NUM_REPL)
  testcase.dfs_train = [t.df_feature for t in trinarys]
  testcase.df_test = testcase.harness.trinary.perturb(
      sigma=SIGMA_TEST)[0].df_feature
  testcase.ser = testcase.harness.trinary.ser_label


class TestMetaClassifier(unittest.TestCase):

  def setUp(self):
    self.harness = HypergridHarness()
    self.df = self.harness.trinary.df_feature
    self.ser = self.harness.trinary.ser_label
    self.dfs = self.makeFeatureDFS(0, SIZE)
    if TEST_LOGISTIC:
      clf = LogisticRegression(random_state=0)
      self.mclf = meta_classifier.MetaClassifierDefault(clf=clf)
    else:
      self.mclf = meta_classifier.MetaClassifierDefault()

  def makeFeatureDFS(self, sigma, size):
    trinary = self.harness.trinary
    trinarys = self.harness.trinary.perturb(sigma, num_repl=size)
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
        self.harness.num_dim)
 
  def testFits(self):
    # Ensure that different classifiers are created
    if IGNORE_TEST:
      return
    trinarys = self.harness.trinary.perturb(sigma=0.3, num_repl=3)
    dfs = [t.df_feature for t in trinarys]
    mclfs = [
        meta_classifier.MetaClassifierAugment(),
        meta_classifier.MetaClassifierAverage(),
        ]
    for mclf in mclfs:
      mclf.fit(dfs, self.ser)
    [m.fit(dfs, self.ser) for m in mclfs]
    self.assertGreater(np.abs(
         mclfs[0].clf.coef_[0][0] - mclfs[1].clf.coef_[0][0]),
         0.1)

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
    score_result = self.mclf.score(self.df, self.ser)
    self.assertEqual(score_result.abs, 1.0)
    self.assertEqual(score_result.rel, 1.0)

  def testLogisticRegression(self):
    if IGNORE_TEST:
      return
    clf = LogisticRegression(random_state=0)
    mclf = meta_classifier.MetaClassifierDefault(clf=clf)
    mclf.fit(self.dfs, self.ser)
    self.assertEqual(len(mclf.clf.coef_[0]),
        self.harness.num_dim)
    score = mclf.score(self.dfs[0], self.ser)
    self.assertEqual(score.abs, SCORE_PERFECT)

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
    setUpGeneric(self)
    self.mclf = meta_classifier.MetaClassifierAverage()

  def testMakeTrainingData(self):
    if IGNORE_TEST:
      return
    testMakeTrainingDataGeneric(self)

  def testScore(self):
    if IGNORE_TEST:
      return
    testScoreGeneric(self)


class TestMetaClassifierAugment(unittest.TestCase):

  def setUp(self):
    setUpGeneric(self)
    self.mclf = meta_classifier.MetaClassifierAugment()

  def testMakeTrainingData(self):
    if IGNORE_TEST:
      return
    testMakeTrainingDataGeneric(self)

  def testScore(self):
    if IGNORE_TEST:
      return
    testScoreGeneric(self)


class TestMetaClassifierEnsemble(unittest.TestCase):

  def setUp(self):
    setUpGeneric(self)
    self.mclf = meta_classifier.MetaClassifierEnsemble()

  def testFit(self):
    if IGNORE_TEST:
      return
    self.mclf.fit(self.dfs_train, self.ser)
    self.assertEqual(len(self.mclf.ensemble), NUM_REPL)

  def testPredict(self):
    if IGNORE_TEST:
      return
    self.mclf.fit(self.dfs_train, self.ser)
    ser = self.mclf.predict(self.df_test)
    self.assertEqual(len(ser), len(self.ser))
    diff = set(ser).symmetric_difference(self.ser)
    self.assertEqual(len(diff), 0)

  def testScore(self):
    if IGNORE_TEST:
      return
    testScoreGeneric(self)

if __name__ == '__main__':
  unittest.main()
