"""Tests for classifier_ensemble.ClassifierEnsemble."""

from common_python.classifier.classifier_ensemble  \
    import ClassifierEnsemble, RandomForestEnsemble
from common_python.testing import helpers
import common_python.constants as cn
from common_python.tests.classifier import helpers as test_helpers

import os
import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = False
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


class TestClassifierEnsemble(unittest.TestCase):
  
  def setUp(self):
    self.df_X, self.ser_y = test_helpers.getData()
    holdouts = 1
    self.ensemble = RandomForestEnsemble(
        self.df_X, self.ser_y, iterations=ITERATIONS)

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
    #
    _ = self.ensemble.plotRank(top=40, title="RandomForest-class 2",
        is_plot=IS_PLOT, class_selection=2)

  def testPlotImportance(self):
    if IGNORE_TEST:
      return
   # Smoke tests
    _ = self.ensemble.plotImportance(top=40, title="RandomForest",
        is_plot=IS_PLOT)
    _ = self.ensemble.plotImportance(top=40, title="RandomForest-class 2", 
        class_selection=2, is_plot=IS_PLOT)


if __name__ == '__main__':
  unittest.main()
