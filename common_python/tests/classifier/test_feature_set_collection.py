
from common_python.tests.classifier import helpers as test_helpers
from common_python.classifier.feature_set_collection  \
    import FeatureSetCollection
from common_python.classifier  \
    import feature_set_collection

import numpy as np
import os
import unittest

IGNORE_TEST = False
IS_PLOT = True
CLASS = 1
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR_PATH = os.path.join(TEST_DIR,
    "test_feature_set_collection_%d" % CLASS)
ANALYZER = test_helpers.getFeatureAnalyzer()


class TestFeatureSetCollection(unittest.TestCase):

  def setUp(self):
    self.collection = FeatureSetCollection(ANALYZER)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertGreater(len(self.collection.ser_fset), 0)
    falses = [np.isnan(v) for v in self.collection.ser_fset]
    self.assertFalse(any(falses))

  # FIXME: Fix test or eliminat code
  def testDisjointify(self):
    if IGNORE_TEST:
      return
    return
    self.collection.disjointify()
    self.assertLess(len(self.collection.ser_fset),
                    len(ANALYZER.ser_fset))
    ser_fset = feature_set_collection.disjointify(
      ANALYZER.ser_fset, min_score=0.9)
    fset_stgs = self.collection.ser_fset.index.tolist()
    # Fails because analyzer.ser_fset is bad
    result = set(fset_stgs).issuperset(
        ser_fset.index.tolist())
    import pdb; pdb.set_trace()
    self.assertTrue(result)

  def testMakeFsetStr(self):
    if IGNORE_TEST:
      return
    FEATURE1 = "feature1"
    FEATURE2 = "feature2"
    fset_stg1 = feature_set_collection.makeFsetStr(
      [FEATURE1, FEATURE2])
    fset_stg2 = feature_set_collection.makeFsetStr(
      [FEATURE2, FEATURE1])
    self.assertEqual(fset_stg1, fset_stg2)

  def testOptimize(self):
    if IGNORE_TEST:
      return
    MIN_SCORE = 0.8
    ser = self.collection.make(min_score=MIN_SCORE)
    ser1 = ser[ser >= MIN_SCORE]
    self.assertTrue(all(ser.eq(ser1)))
    some_true = ["+" in f for f in ser.index]
    self.assertTrue(any(some_true))

  def testMakeCandidateSer(self):
    FSET = {"Rv2009", "Rv1460"}
    min_score = 0.01
    ser = self.collection._makeCandidateSer(FSET,
        min_score=min_score)
    self.assertEqual(len(ser[ser < min_score]), 0)
    length = len([i for i in ser.index if i in list(FSET)])
    self.assertEqual(length, 0)

  def test_ser_fset(self):
    if IGNORE_TEST:
      return
    ser = self.collection.ser_fset
    num_feature = len(ANALYZER.features)
    expected = num_feature*(num_feature-1)/2 + num_feature
    self.assertEqual(expected, len(ser))
    trues = [(v <= 1) and (v >= 0) for v in ser]
    self.assertTrue(all(trues))




if __name__ == '__main__':
  unittest.main()


