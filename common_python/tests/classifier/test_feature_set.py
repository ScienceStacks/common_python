from common_python.tests.classifier import helpers as test_helpers
from common_python.classifier.feature_set  \
    import FeatureSet
from common_python.testing import helpers
from common_python import constants as cn

import numpy as np
import os
import unittest

IGNORE_TEST = False
IS_PLOT = False
CLASS = 1
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR_PATH = os.path.join(TEST_DIR,
    "test_feature_set_collection_%d" % CLASS)
ANALYZER = test_helpers.getFeatureAnalyzer()
FEATURE1 = "feature1"
FEATURE2 = "feature2"
FEATURE_SET_STG = "Rv2009+Rv3830c"


class TestFeatureSet(unittest.TestCase):

  def setUp(self):
    self.fset = FeatureSet([FEATURE1, FEATURE2])

  def testStr(self):
    if IGNORE_TEST:
      return
    self.assertEqual(self.fset.str, str(self.fset))

  def testEquals(self):
    if IGNORE_TEST:
      return
    fset = FeatureSet([FEATURE2, FEATURE1])
    self.assertTrue(self.fset.equals(fset))

  def testProfileInstance(self):
    if IGNORE_TEST:
      return
    fset = FeatureSet(FEATURE_SET_STG,
        analyzer=ANALYZER)
    df = fset.profileInstance()
    columns = [cn.SUM, cn.PREDICTED, cn.CLASS,
        cn.INTERCEPT]
    columns.extend(fset.list)
    self.assertTrue(helpers.isValidDataFrame(df,
        expected_columns=columns))

  def testplotProfileInstance(self):
    if IGNORE_TEST:
      return
    # Smoke test
    fset = FeatureSet(FEATURE_SET_STG, analyzer=ANALYZER)
    fset.plotProfileInstance(is_plot=IS_PLOT)

  def testplotProfileTrinary(self):
    if IGNORE_TEST:
      return
    fset = FeatureSet(FEATURE_SET_STG, analyzer=ANALYZER)
    df = fset.profileTrinary()
    columns = [cn.PREDICTED, cn.FRAC, cn.SIGLVL_POS,
        cn.SIGLVL_NEG, cn.COUNT, cn.FEATURE_SET]
    self.assertTrue(helpers.isValidDataFrame(df,
        expected_columns=columns))


if __name__ == '__main__':
  unittest.main()


