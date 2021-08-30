import common_python.constants as cn
from common_python.classifier.feature_set  \
    import FeatureSet, FeatureVector
from common_python.classifier.cc_case.case_core \
    import  Case, FeatureVectorStatistic
from common_python.classifier.cc_case.case_builder import CaseBuilder
from common_python import constants as cn
from common.trinary_data import TrinaryData
from common_python.tests.classifier.cc_case import helpers

import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import unittest

IGNORE_TEST = False
IS_PLOT = False
CLASS = 1
DF_X, SER_Y_ALL = helpers.getData()
SER_Y = SER_Y_ALL.apply(lambda v: 1 if v == CLASS else 0)
FEATURE_A = "Rv2009"
FEATURE_B = "Rv3830c"
FEATURE_SET_STG = FEATURE_A + "+" + FEATURE_B
VALUE1 = -1
VALUE2 = -1 
VALUES = [VALUE1, VALUE2]
FEATURE_VECTOR = FeatureVector(
    {f: v for f,v in zip([FEATURE_A, FEATURE_B], VALUES)})
COUNT = 10
FRAC = 0.2
SIGLVL = 0.05


class TestFeatureVectorStatistic(unittest.TestCase):

  def setUp(self):
    self.statistic = FeatureVectorStatistic(COUNT, int(FRAC*COUNT), 0.5, SIGLVL)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(self.statistic.num_sample, COUNT)
    self.assertNotEqual(self.statistic.num_sample, COUNT - 1)

  def testEq(self):
    if IGNORE_TEST:
      return
    self.assertTrue(self.statistic == self.statistic)
    statistic = FeatureVectorStatistic(COUNT + 1, int(FRAC*COUNT), 0.5, SIGLVL)
    self.assertFalse(statistic == self.statistic)


class TestCase(unittest.TestCase):

  def setUp(self):
    self.fv_statistic = FeatureVectorStatistic(COUNT,
        int(FRAC*COUNT), 0.5, SIGLVL)
    self.case = Case(FEATURE_VECTOR, self.fv_statistic, df_X=DF_X)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(self.case.fv_statistic.num_sample, COUNT)
    self.assertGreater(len(self.case.instance_str), 0)

  def testEq(self):
    if IGNORE_TEST:
      return
    self.assertTrue(self.case == self.case)
    fv_statistic = FeatureVectorStatistic(COUNT - 1, int(FRAC*COUNT), 0.5,
        SIGLVL)
    case = Case(FEATURE_VECTOR, fv_statistic)
    self.assertFalse(case == self.case)

  def testGetCompatibleInstances(self):
    if IGNORE_TEST:
      return
    result = Case._getCompatibleInstances(DF_X, FEATURE_VECTOR)
    self.assertGreater(len(list(result)), 0)
    self.assertTrue(result[0] in DF_X.index)


if __name__ == '__main__':
  unittest.main()
