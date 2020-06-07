
from common_python.tests.classifier import helpers as test_helpers
from common_python.classifier.feature_set_analyzer import FeatureSetAnalyzer
from common_python.classifier import feature_set_analyzer

import numpy as np
import os
import unittest

IGNORE_TEST = False
IS_PLOT = True
CLASS = 1
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR_PATH = os.path.join(TEST_DIR,
                           "test_feature_analyzer_%d" % CLASS)
ANALYZER = test_helpers.getFeatureAnalyzer()


class TestFeatureSet(unittest.TestCase):

  def setUp(self):
    self.fs_analyzer = FeatureSetAnalyzer(ANALYZER)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertGreater(len(self.fs_analyzer.ser_fset), 0)
    falses = [np.isnan(v) for v in self.fs_analyzer.ser_fset]
    self.assertFalse(any(falses))

  # FIXME: Fix test or eliminat code
  def testDisjointify(self):
    if IGNORE_TEST:
      return
    return
    self.fs_analyzer.disjointify()
    self.assertLess(len(self.fs_analyzer.ser_fset),
                    len(ANALYZER.ser_fset))
    ser_fset = feature_set_analyzer.disjointify(
      ANALYZER.ser_fset, min_score=0.9)
    fset_stgs = self.fs_analyzer.ser_fset.index.tolist()
    # Fails because analyzer.ser_fset is bad
    result = set(fset_stgs).issuperset(
        ser_fset.index.tolist())
    import pdb; pdb.set_trace()
    self.assertTrue(result)

  def testOptimize(self):
    if IGNORE_TEST:
      return
    MIN_SCORE = 0.8
    ser = self.fs_analyzer.optimize(min_score=MIN_SCORE)
    ser1 = ser[ser >= MIN_SCORE]
    self.assertTrue(all(ser.eq(ser1)))
    some_true = ["+" in f for f in ser.index]
    self.assertTrue(any(some_true))



if __name__ == '__main__':
  unittest.main()


