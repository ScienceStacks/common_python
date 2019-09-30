"""Tests for classifier_ensemble.ClassifierEnsemble."""

from common_python.classifier import classifier_ensemble
from common_python.testing import helpers
import common_python.constants as cn
from common.trinary_data import TrinaryData
from common_python.classifier.svm_ensemble import SVMEnsemble

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


class TestSVMEnsemble(unittest.TestCase):

  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()

  def _init(self):
    self.df_X, self.ser_y = getData()
    self.ensemble = SVMEnsemble(svm.LinearSVC(), size=SIZE)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertTrue(isinstance(self.ensemble.base_clf, svm.LinearSVC))

  # TODO: Test training with featuers
  def testFit(self):
    if IGNORE_TEST:
      return
    self._init()
    self.ensemble.fit(self.df_X, self.ser_y)
    for items in [self.ensemble.clfs, self.ensemble.scores]:
      self.assertEqual(len(items), SIZE)

  def testOrderFeatures(self):
    if IGNORE_TEST:
      return
    clf = self.ensemble.clfs[0]
    import pdb; pdb.set_trace()
    result = self.ensemble._orderFeatures(clf, None)
    self.assertEqual(len(result), len(self.ensemble.features))
    #
    result = self.ensemble._orderFeatures(clf, 1)
    self.assertEqual(len(result), len(self.ensemble.features))

  def testMakeRankDF(self):
    if IGNORE_TEST:
      return
    df = self.ensemble.makeRankDF()
    self.assertTrue(helpers.isValidDataFrame(df,
        [cn.MEAN, cn.STD, cn.STERR]))

  def testMakeImportanceDF(self):
    if IGNORE_TEST:
      return
    df = self.ensemble.makeImportanceDF()
    self.assertTrue(helpers.isValidDataFrame(df,
        [cn.MEAN, cn.STD, cn.STERR]))
    #
    df1 = self.ensemble.makeImportanceDF(class_selection=1)
    self.assertTrue(helpers.isValidDataFrame(df,
        [cn.MEAN, cn.STD, cn.STERR]))
    trues = [df.loc[i, cn.MEAN] >= df1.loc[i, cn.MEAN] 
        for i in df.index]
    self.assertTrue(all(trues))

  def testPlotRank(self):
    if IGNORE_TEST:
      return
   # Smoke tests
    _ = self.ensemble.plotRank(top=40, title="SVM", is_plot=IS_PLOT)
    #
    _ = self.ensemble.plotRank(top=40, title="SVM-class 2",
        is_plot=IS_PLOT, class_selection=2)

  def testPlotImportance(self):
    if IGNORE_TEST:
      return
   # Smoke tests
    _ = self.ensemble.plotImportance(top=40, title="SVM",
        is_plot=IS_PLOT)
    _ = self.ensemble.plotImportance(top=40, title="SVM-class 2", 
        class_selection=2, is_plot=IS_PLOT)


if __name__ == '__main__':
  unittest.main()
