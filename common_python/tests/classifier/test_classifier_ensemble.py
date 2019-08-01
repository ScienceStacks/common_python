"""Tests for classifier_ensemble.ClassifierEnsemble."""

from common_python.classifier import classifier_ensemble
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
  df_X = DATA.df_X.drop(index="T1")
  df_X.columns = DATA.features
  ser_y = DATA.ser_y.drop(index="T1")
  return df_X, ser_y

class TestClassifierEnsemble(unittest.TestCase):

  def setUp(self):
    self.lin_clf = svm.LinearSVC()
    self.cls = classifier_ensemble.ClassifierEnsemble
    self.ensemble = self.cls([], [], [])

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(len(self.ensemble.classifiers), 0)
    self.assertEqual(len(self.ensemble.features), 0)
    self.assertEqual(len(self.ensemble.classes), 0)

  def testCrossVerify(self):
    def test(holdouts, kwargs=None):
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
          result = classifier_ensemble.LinearSVMEnsemble.crossValidate(
              DF, SER, classifier_args=kwargs,
              iterations=10, holdouts=holdouts)
          self.assertTrue(isinstance(result, 
              classifier_ensemble.CrossValidationResult))
        except ValueError:
          raise ValueError
    #
    test(1)
    with self.assertRaises(ValueError):
      test(2)
      pass
    test(1, kwargs={"C": 0.5, "penalty": "l2"})

  def testDo1(self):
    if IGNORE_TEST:
      return
    df_X, ser_y = getData()
    holdouts = 1
    result = classifier_ensemble.LinearSVMEnsemble.crossValidate(
        df_X, ser_y, iterations=10, holdouts=holdouts)
    self.assertEqual(len(df_X.columns), 
        len(result.ensemble.features))


class TestLinearSVMEnsemble(unittest.TestCase):

  def setUp(self):
    self.lin_clf = svm.LinearSVC()
    self.cls = classifier_ensemble.LinearSVMEnsemble
    df_X, ser_y = getData()
    holdouts = 1
    result = self.cls.crossValidate(df_X, ser_y, 
        iterations=100, holdouts=holdouts)
    self.ensemble = self.cls(result.ensemble.classifiers,
        df_X.columns.tolist(), ser_y.index.tolist())

  def testCrossVerify(self):
    if IGNORE_TEST:
      return
    def test(holdouts):
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
          result = self.cls.crossValidate(DF, SER, 
              iterations=10, holdouts=holdouts)
          self.assertTrue(isinstance(result, 
              classifier_ensemble.CrossValidationResult))
        except ValueError:
          raise ValueError
    #
    test(1)
    with self.assertRaises(ValueError):
      test(2)
      pass

  def testOrderFeatures(self):
    if IGNORE_TEST:
      return
    clf = self.ensemble.classifiers[0]
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


class TestRandomForestEnsemble(unittest.TestCase):

  def setUp(self):
    df_X, ser_y = getData()
    self.cls = classifier_ensemble.RandomForestEnsemble
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
