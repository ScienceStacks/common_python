"""Tests for classifier_ensemble.ClassifierEnsemble."""

from common_python.classifier.classifier_ensemble_random_forest  \
    import ClassifierEnsembleRandomForest
from common_python.testing import helpers
import common_python.constants as cn
from common.trinary_data import TrinaryData

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import unittest
import warnings

IGNORE_TEST = True
IS_PLOT = False
SIZE = 10
values = list(range(SIZE))
values.extend(values)
DF = pd.DataFrame({
    'A': values,
    'B': np.repeat(1, 2*SIZE),
    })
SER = pd.Series(values)
DATA = TrinaryData()


def getData():
  df_X = DATA.df_X
  df_X.columns = DATA.features
  ser_y = DATA.ser_y
  return df_X, ser_y


class TestClassifierEnsembleRandomForest(unittest.TestCase):

  def setUp(self):
    df_X, ser_y = getData()
    self.cls = ClassifierEnsembleRandomForest
    self.ensemble = self.cls(df_X, ser_y)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertTrue(isinstance(self.ensemble.random_forest,
        RandomForestClassifier))

  def testMakeRankDF(self):
    if IGNORE_TEST:
      return
    df = self.ensemble.makeRankDF()
    self.assertTrue(helpers.isValidDataFrame(df,
        [cn.MEAN, cn.STD, cn.STERR]))

  def testPlotRank(self):
    if IGNORE_TEST:
      return
   # Smoke tests
    _ = self.ensemble.plotRank(top=40, title="RandomForest", is_plot=IS_PLOT)

  def testPlotImportance(self):
    if IGNORE_TEST:
      return
   # Smoke tests
    _ = self.ensemble.plotImportance(top=40, title="RandomForest", is_plot=IS_PLOT)

  def testMakeImportanceDF(self):
    if IGNORE_TEST:
      return
    df = self.ensemble.makeImportanceDF()
    self.assertTrue(helpers.isValidDataFrame(df,
        [cn.MEAN, cn.STD, cn.STERR]))


if __name__ == '__main__':
  unittest.main()
