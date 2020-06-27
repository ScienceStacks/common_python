from common_python.tests.classifier import helpers as test_helpers
from common_python.classifier.feature_set  \
    import FeatureSet
from common_python.classifier import feature_set
from common_python.testing import helpers
from common_python import constants as cn

import copy
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
DF_X = ANALYZER.df_X
SER_Y = ANALYZER.ser_y
sorted_index = sorted(DF_X.index.tolist(),
     key=feature_set.SORT_FUNC)
DF_X = DF_X.loc[sorted_index, :]
SER_Y = SER_Y.loc[sorted_index]
FEATURE1 = "feature1"
FEATURE2 = "feature2"
FEATURE_SET_STG = "Rv2009+Rv3830c"


class TestFeatureSet(unittest.TestCase):

  def setUp(self):
    self.fset = FeatureSet([FEATURE1, FEATURE2])
    self.df_X = copy.deepcopy(DF_X)
    self.ser_y = copy.deepcopy(SER_Y)

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

  def testEvaluate(self):
    if IGNORE_TEST:
      return
    fset = FeatureSet(FEATURE_SET_STG, analyzer=ANALYZER)
    ser = fset.evaluate(self.df_X,
        is_include_neg=False)
    self.assertEqual(len(ser), len(self.df_X))
    cls_1 = self.ser_y.index[self.ser_y == 1]
    cls_0 = self.ser_y.index[self.ser_y == 0]
    # Should have higher significance levels for
    # negative class than for positive class.
    self.assertGreater(ser.loc[cls_0].mean(),
        ser.loc[cls_1].mean())

  def testEvaluate2(self):
    if IGNORE_TEST:
      return
    fset = FeatureSet(FEATURE_SET_STG, analyzer=ANALYZER)
    ser_include = fset.evaluate(self.df_X,
        is_include_neg=True)
    num_pos = sum([1 for v in ser_include if v < 0])
    self.assertGreater(num_pos, 0)


if __name__ == '__main__':
  unittest.main()


