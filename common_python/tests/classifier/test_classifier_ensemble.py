"""Tests for classifier_ensemble.ClassifierEnsemble."""

from common_python.classifier.classifier_ensemble  \
    import ClassifierEnsemble, RandomForestEnsemble
from common_python.classifier.classifier_collection  \
    import ClassifierCollection
from common_python.testing import helpers
import common_python.constants as cn
from common_python.tests.classifier import helpers as test_helpers

import collections
import os
import pandas as pd
import random
import numpy as np
import unittest

IGNORE_TEST = True
IS_PLOT = False
SIZE = 10
ITERATIONS = 3
values = list(range(SIZE))
values.extend(values)
DF = pd.DataFrame({
    'A': values,
    'B': np.repeat(1, 2*SIZE),
    })
SER = pd.Series(values)


######## Helper Classes ######
class RandomClassifier(object):

  def __init__(self):
    self.class_probs = None

  def fit(self, _, ser_y):
    """
    Fits by recording probability of each class.
    :param object _:
    :param pd.Series ser_y:
    """
    class_counts = dict(collections.Counter(ser_y.values.tolist()))
    self.class_probs = {k: v / len(ser_y) 
         for k,v in class_counts.items()}

  def predict(self, df_X):
    """
    :param pd.DataFrame df_X:
    :return pd.Series:
    """
    randoms = np.random.randint(0, len(self.class_probs.keys()),
        len(df_X))
    values = [list(self.class_probs.keys())[n] for n in randoms]
    ser = pd.Series(values, index=df_X.index)
    return ser

  def score(self, df_X, ser_y):
    ser = self.predict(df_X)
    value = np.mean(ser == ser_y)
    return value


######## Test Classes ######
class TestClassifierEnsemble(unittest.TestCase):

  def init(self):
    self.df_X, self.ser_y = test_helpers.getData()
    holdouts = 1
    self.ensemble = RandomForestEnsemble(
        self.df_X, self.ser_y, iterations=ITERATIONS)
    self.random_classifier_ensemble =  \
      ClassifierEnsemble.makeByRandomHoldout(
      RandomClassifier(), DF, SER, 100, 1)
  
  def setUp(self):
    if IGNORE_TEST:
      return
    self.init()

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertGreater(len(self.ensemble.clfs), 0)
    self.assertGreater(len(self.ensemble.features), 0)
    self.assertGreater(len(self.ensemble.classes), 0)

  def testMakeRankDF(self):
    if IGNORE_TEST:
      return
    df = self.ensemble.makeRankDF()
    self.assertTrue(helpers.isValidDataFrame(df,
        [cn.MEAN, cn.STD, cn.STERR]))

  def testMakeImportanceDF(self):
    if IGNORE_TEST:
      return
    def test(df):
      self.assertTrue(helpers.isValidDataFrame(df,
          [cn.MEAN, cn.STD, cn.STERR]))
      trues = [df.loc[i, cn.STD] >= df.loc[i, cn.STERR] 
          for i in df.index]
      self.assertTrue(all(trues))
    #
    df = self.ensemble.makeImportanceDF()
    test(df)
    #
    ensemble = RandomForestEnsemble(self.df_X, self.ser_y,
        iterations=10)
    df1 = ensemble.makeImportanceDF()
    test(df1)

  def testPlotRank(self):
    if IGNORE_TEST:
      return
   # Smoke tests
    _ = self.ensemble.plotRank(top=40, title="RandomForest", is_plot=IS_PLOT)

  def testPlotImportance(self):
    if IGNORE_TEST:
      return
   # Smoke tests
    _ = self.ensemble.plotImportance(top=40, title="RandomForest",
        is_plot=IS_PLOT)

  def testRandomClassifier(self):
    if IGNORE_TEST:
      return
    clf = RandomClassifier()
    clf.fit(DF, SER)
    self.assertEqual(clf.class_probs[0], 1/SIZE)
    ser_predict = clf.predict(DF)
    trues = [c in range(SIZE) for c in ser_predict.values]
    self.assertTrue(all(trues))

  def testPredict(self):
    if IGNORE_TEST:
      return
    ser = self.random_classifier_ensemble.predict(DF.loc[0,:])
    mean = ser.mean(axis=1).values[0]
    expected = 1/SIZE
    self.assertLess(abs(mean - expected), 0.1)

  def testScore(self):
    self.init()
    score = self.random_classifier_ensemble.score(DF, SER)
    expected = 1/SIZE
    self.assertLess(abs(score- expected), 0.1)


if __name__ == '__main__':
  unittest.main()
