from common_python.classifier.feature_set  \
    import FeatureSet, FeatureVector
from common_python.classifier.case_manager import CaseManager, Case
from common_python import constants as cn
from common_python.tests.classifier import helpers as test_helpers

import copy
import numpy as np
import os
import pandas as pd
import unittest

IGNORE_TEST = True
IS_PLOT = True
CLASS = 1
ANALYZER = test_helpers.getFeatureAnalyzer()
DF_X = ANALYZER.df_X
SER_Y = ANALYZER.ser_y
SER_Y = SER_Y.apply(lambda v: 1 if v == CLASS else 0)
FEATURE_A = "Rv2009"
FEATURE_B = "Rv3830c"
FEATURE_SET_STG = FEATURE_A + "+" + FEATURE_B
VALUE1 = 1
VALUE2 = -1 
VALUES = [VALUE1, VALUE2]
FEATURE_VECTOR = FeatureVector(
    {f: v for f,v in zip([FEATURE_A, FEATURE_B], VALUES)})
COUNT = 10
FRAC = 0.2
SIGLVL = 0.05


class TestCase(unittest.TestCase):

  def setUp(self):
    fv_statistic = FeatureVectorStatistic(sl=SIGLVL, cnt=COUNT,
        pos=int(FRAC*COUNT))
    self.case = Case(FEATURE_VECTOR, fv_statistic)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(self.case.fv_statistic.count, COUNT)


class TestCaseManager(unittest.TestCase):

  def setUp(self):
    self.manager = CaseManager(DF_X, SER_Y)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertGreater(len(self.manager._features), 0)

  def testGetCompatibleFeatureValues(self):
    if IGNORE_TEST:
      return
    values = self.manager._getCompatibleFeatureValues(FEATURE_B, 1)
    self.assertEqual(values, [-1, 0, 1])
    values = self.manager._getCompatibleFeatureValues(FEATURE_B, 0.5)
    self.assertEqual(values, [-1, 0])

  def testGetFeatureVectors(self):
    if IGNORE_TEST:
      return
    dtree = self.manager._forest.estimators_[0]
    feature_vectors = self.manager._getFeatureVectors(dtree)
    import pdb; pdb.set_trace()

  def testGetStatistic(self):
    # TESTING
    statistic = self.manager.getFeatureVectorStatistic(FEATURE_VECTOR)
    import pdb; pdb.set_trace()
    self.assertLess(statistic.sl, 0)
    self.assertLess(np.abs(statistic.sl), 0.01)
    self.assertGreater(statistic.cnt, statistic.pos)


if __name__ == '__main__':
  unittest.main()

