"""Tests for CrossValidator."""

from common_python.classifier.cross_validator import CrossValidator
import common_python.constants as cn
#from common.trinary_data import TrinaryData

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


class TestCrossValidator(unittest.TestCase):

  def setUp(self):
    self.lin_clf = svm.LinearSVC()
    self.cross = CrossValidator(self.lin_clf, DF, SER)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertTrue(self.cross.df_X.equals(DF))
    self.assertTrue(self.cross.ser_y.equals(SER))

  def testDo(self):
    if IGNORE_TEST:
      return
    def test(holdouts):
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
          result = self.cross.do(iterations=10, holdouts=holdouts)
          self.assertEqual(len(result), 2)
          self.assertTrue(np.isclose(result[1], 0))
        except ValueError:
          raise ValueError
    #
    test(1)
    with self.assertRaises(ValueError):
      test(2)
      pass

#  def testDo1(self):
#    data = TrinaryData()
#    cross = CrossValidator(self.lin_clf, data.df_X, data.ser_y)
#    result = cross.do(iterations=500)
#    import pdb; pdb.set_trace()

if __name__ == '__main__':
  unittest.main()
