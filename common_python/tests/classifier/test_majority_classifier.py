from common_python.classifier import majority_classifier
from common_python.testing import helpers
import common_python.constants as cn
from common_python.tests.classifier import helpers as test_helpers

import pandas as pd
import numpy as np
from sklearn import svm
import unittest

IGNORE_TEST = False
SIZE = 30
COL_A = 'a'
COL_B = 'b'
DF = pd.DataFrame({
    COL_A: range(SIZE)})
DF[COL_B] = 10*DF[COL_A]
LABEL_X = "X"
LABEL_Y = "Y"
LABEL_Z = "Z"
X_INT = 5
Y_INT = 10
Z_INT = 15
SER = pd.Series(np.concatenate([
    np.repeat(LABEL_X, X_INT),
    np.repeat(LABEL_Y, Y_INT),
    np.repeat(LABEL_Z, Z_INT),
    ]))


class TestMajorityClassifier(unittest.TestCase):

  def setUp(self):
    self.clf = majority_classifier.MajorityClassifier()

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertIsNone(self.clf.majority)

  def testFit(self):
    if IGNORE_TEST:
      return
    self.clf.fit(DF, SER)
    self.assertEqual(self.clf.majority, LABEL_Z)
    self.assertEqual(str(self.clf.majority), LABEL_Z)

  def testPredict(self):
    if IGNORE_TEST:
      return
    self.clf.fit(DF, SER)
    ser = self.clf.predict(DF)
    ser_expected = pd.Series(np.repeat(LABEL_Z, SIZE))
    self.assertTrue(ser.equals(ser_expected))

  def testScore(self):
    if IGNORE_TEST:
      return
    self.clf.fit(DF, SER)
    accuracy = self.clf.score(DF, SER)
    expected = Z_INT / SIZE
    self.assertTrue(accuracy, expected)


if __name__ == '__main__':
  unittest.main()
