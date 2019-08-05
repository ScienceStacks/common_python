"""Tests for classifier utilities."""

from common_python.classifier.classifier_collection  \
    import ClassifierCollection
from common_python.testing import helpers
import common_python.constants as cn
from common.trinary_data import TrinaryData

import pandas as pd
import numpy as np
from sklearn import svm
import unittest
import warnings

IGNORE_TEST = True
SIZE = 10
values = list(range(SIZE))
values.extend(values)
DF = pd.DataFrame({
    'A': values,
    'B': np.repeat(1, 2*SIZE),
    })
SER = pd.Series(values)
DATA = TrinaryData()
EMPTY_LIST = []


def getData():
  df_X = DATA.df_X
  df_X.columns = DATA.features
  ser_y = DATA.ser_y
  return df_X, ser_y

class TestClassifierCollection(unittest.TestCase):

  def setUp(self):
    self.clf = svm.LinearSVC()
    self.lin_clf = svm.LinearSVC()

  def testConstructor(self):
    if IGNORE_TEST:
      return
    collection = ClassifierCollection(EMPTY_LIST,
        EMPTY_LIST, EMPTY_LIST)
    for item in ["clfs", "features", "classes"]:
      statement = "isinstance(%s, list)" % item
      self.assertTrue(statement)

  def testMakeByRandomStateHoldout(self):
    if IGNORE_TEST:
      return
    def test(holdouts):
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
          result = ClassifierCollection.crossVerifyByState(
              self.clf, DF, SER, 10, holdouts=holdouts)
          for value in [result.mean, result.std]:
            self.assertTrue(isinstance(value, float))
          self.assertTrue(isinstance(result.collection,
              ClassifierCollection))
        except ValueError:
          raise ValueError
    #
    test(1)
    with self.assertRaises(ValueError):
      test(len(DF))
      pass

  def testMakeByRandomStateHoldout2(self):
    df_X, ser_y = getData()
    holdouts = 1
    result = ClassifierCollection.crossVerifyByState(
        self.clf, df_X, ser_y, num_clfs=100, holdouts=holdouts)
    self.assertEqual(len(df_X.columns), 
        len(result.collection.features))
    self.assertGreater(result.mean, 0.95)


if __name__ == '__main__':
  unittest.main()
