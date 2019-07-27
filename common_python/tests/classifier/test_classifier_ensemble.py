"""Tests for classifier_ensemble.ClassifierEnsemble."""

from common_python.classifier import classifier_ensemble
import common_python.constants as cn
from common.trinary_data import TrinaryData

import pandas as pd
import numpy as np
from sklearn import svm
import unittest
import warnings

IGNORE_TEST = False
SIZE = 10
values = list(range(SIZE))
values.extend(values)
DF = pd.DataFrame({
    'A': values,
    'B': np.repeat(1, 2*SIZE),
    })
SER = pd.Series(values)


class TestClassifierEnsemble(unittest.TestCase):

  def setUp(self):
    self.lin_clf = svm.LinearSVC()
    self.cls = classifier_ensemble.ClassifierEnsemble
    self.ensemble = classifier_ensemble.ClassifierEnsemble([])

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(len(self.ensemble.classifiers), 0)

  def testCrossVerify(self):
    if IGNORE_TEST:
      return
    def test(holdouts):
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
          result = self.cls.crossVerify(
              self.lin_clf, DF, SER, 
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

  def testDo1(self):
    data = TrinaryData()
    df_X = data.df_X.drop(index="T1")
    ser_y = data.ser_y.drop(index="T1")
    holdouts = 1
    result = self.cls.crossVerify(
        self.lin_clf, df_X, ser_y, 
        iterations=200, holdouts=holdouts)

if __name__ == '__main__':
  unittest.main()
