from common_python.tests.classifier import helpers as test_helpers
from common_python.classifier import feature_analyzer
from common_python.classifier.feature_set  \
    import FeatureSet, FeatureVector
from common_python.classifier import feature_set
from common_python.testing import helpers
from common_python import constants as cn

import copy
import numpy as np
import os
import pandas as pd
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
FEATURE3 = "feature3"
FEATURE_SET_STG = "Rv2009+Rv3830c"
VALUE1 = 1
VALUE2 = -1 
VALUE3 = 10
VALUES = [VALUE1, VALUE2]


class TestFeatureVector(unittest.TestCase):

  def setUp(self):
    self.fset = FeatureSet([FEATURE1, FEATURE2])
    self.feature_vector = FeatureVector(self.fset, VALUES)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    dct = {FEATURE1: VALUE1, FEATURE2: VALUE2}
    feature_vector = FeatureVector(self.fset, dct)
    self.assertTrue(feature_vector.equals(
        self.feature_vector))
    #
    feature_vector2 = FeatureVector(dct)
    feature_vector.equals(feature_vector2)
    #
    feature_vector2 = FeatureVector(pd.Series(dct))
    feature_vector.equals(feature_vector2)

  def testIsSubvector(self):
    if IGNORE_TEST:
      return
    def test(features, values, expected=True):
      dct = {k: v for k, v in zip(features, values)}
      feature_vector = FeatureVector(dct)
      result = self.feature_vector.isSubvector(feature_vector)
      self.assertEqual(result, expected)
    #
    test([FEATURE2, FEATURE1], [VALUE2, VALUE1])
    test([FEATURE1, FEATURE2], [VALUE1, VALUE2])
    test([FEATURE1], [VALUE1])
    test([FEATURE1], [2], expected=False)

  def testIsCompatible(self):
    if IGNORE_TEST:
      return
    dct1 = {FEATURE1: VALUE1, FEATURE2: VALUE2}
    dct2 = {FEATURE1: VALUE1, FEATURE3: VALUE3}
    dct3 = {FEATURE3: VALUE3}
    def test(dct1, dct2):
      fset1 = FeatureSet(dct1.keys())
      fset2 = FeatureSet(dct2.keys())
      feature_vector1 = FeatureVector(fset1, dct1)
      feature_vector2 = FeatureVector(fset2, dct2)
      result = feature_vector1.isCompatible(
          feature_vector2)
      return result
    #
    self.assertTrue(test(dct1, dct1))
    self.assertTrue(test(dct1, dct2))
    self.assertFalse(test(dct1, dct3))


  def testIsCompatible(self):
    if IGNORE_TEST:
      return
    dct1 = {FEATURE1: VALUE1, FEATURE2: VALUE2}
    dct2 = {FEATURE1: VALUE1, FEATURE3: VALUE3}
    dct3 = {FEATURE3: VALUE3}
    def test(dct1, dct2):
      fset1 = FeatureSet(dct1.keys())
      fset2 = FeatureSet(dct2.keys())
      feature_vector1 = FeatureVector(fset1, dct1)
      feature_vector2 = FeatureVector(fset2, dct2)
      result = feature_vector1.isCompatible(
          feature_vector2)
      return result
    #
    self.assertTrue(test(dct1, dct1))
    self.assertTrue(test(dct1, dct2))
    self.assertFalse(test(dct1, dct3))

  def testMake(self):
    if IGNORE_TEST:
      return
    feature_vector= FeatureVector.make(
        str(self.feature_vector))
    self.assertTrue(feature_vector.equals(
        self.feature_vector))



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
    df = fset.evaluate(self.df_X,
        is_include_neg=False)
    ser = df[cn.SIGLVL]
    self.assertEqual(len(df), len(self.df_X))
    cls_1 = self.ser_y.index[self.ser_y == 1]
    cls_0 = self.ser_y.index[self.ser_y == 0]
    # Should have higher significance levels for
    # negative class than for positive class.
    self.assertGreater(df[cn.SIGLVL].loc[cls_0].mean(),
        ser.loc[cls_1].mean())

  def testEvaluate2(self):
    if IGNORE_TEST:
      return
    fset = FeatureSet(FEATURE_SET_STG, analyzer=ANALYZER)
    ser_include = fset.evaluate(self.df_X,
        is_include_neg=True)[cn.SIGLVL]
    num_pos = sum([1 for v in ser_include if v < 0])
    self.assertGreater(num_pos, 0)

  def testGetFeatureVector(self):
    if IGNORE_TEST:
      return
    value1 = 1
    value2 = 2
    value3 = 3
    values = [value1, value2, value3]
    ser_X = pd.Series(values)
    ser_X.index = [FEATURE1, FEATURE2, FEATURE3]
    #
    fset_stg = "%s%s%s%s%s" % (FEATURE1,
        cn.FEATURE_SEPARATOR, FEATURE2,
        cn.FEATURE_SEPARATOR, FEATURE3)
    fset = FeatureSet(fset_stg)
    vector = fset.getFeatureVector(ser_X)
    other_vector= FeatureVector(fset, 
       (value1, value2, value3))
    self.assertTrue(vector.equals(other_vector))
    #
    fset_stg = "%s%s%s%s%s" % (FEATURE1,
        feature_analyzer.SEPARATOR, FEATURE2,
        cn.FEATURE_SEPARATOR, FEATURE3)
    fset = FeatureSet(fset_stg)
    vector = fset.getFeatureVector(ser_X)
    expected_vector = FeatureVector(fset,
        [value1, value3])
    self.assertTrue(vector.equals(expected_vector))


if __name__ == '__main__':
  unittest.main()


