from common_python.classifier.feature_set  \
    import FeatureSet, FeatureVector
from common_python.classifier.case_manager  \
    import CaseManager, Case, FeatureVectorStatistic
from common_python import constants as cn
from common_python.tests.classifier import helpers as test_helpers

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
ANALYZER = test_helpers.getFeatureAnalyzer()
DF_X = ANALYZER.df_X
SER_Y_ALL = ANALYZER.ser_y
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


class TestCase(unittest.TestCase):

  def setUp(self):
    fv_statistic = FeatureVectorStatistic(COUNT, int(FRAC*COUNT), SIGLVL)
    self.case = Case(FEATURE_VECTOR, fv_statistic)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(self.case.fv_statistic.num_sample, COUNT)


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

  def testGetStatistic(self):
    if IGNORE_TEST:
      return
    statistic = self.manager._getFeatureVectorStatistic(FEATURE_VECTOR)
    self.assertLess(statistic.siglvl, 0)
    self.assertLess(np.abs(statistic.siglvl), 0.05)
    self.assertGreater(statistic.num_sample, statistic.num_pos)

  def testGetCases(self):
    if IGNORE_TEST:
      return
    clf = RandomForestClassifier(max_depth=4, random_state=0, bootstrap=False,
                                min_impurity_decrease=.01, min_samples_leaf=5)
    _ = clf.fit(DF_X, SER_Y)
    dtree = clf.estimators_[2]
    case_dct = self.manager._getCases(dtree)
    self.assertTrue(len(case_dct) > 3)
    if IS_PLOT:
      self.manager.displayCases(cases=case_dct.values())
      plt.show()

  def testBuild(self):
    if IGNORE_TEST:
      return
    num_tree = 20
    manager = CaseManager(DF_X, SER_Y, n_estimators=num_tree)
    manager.build()
    self.assertGreater(len(manager.case_dct), 2*num_tree)

  def testPlotEvaluate(self):
    if IGNORE_TEST:
      return
    num_tree = 10
    manager = CaseManager(DF_X, SER_Y, n_estimators=num_tree)
    manager.build()
    ser_X = DF_X.loc["T14.0", :]
    cases = manager.plotEvaluate(ser_X,
        title="State 1 evaluation for T14.0", is_plot=IS_PLOT)
    ser_X = DF_X.loc["T2.0", :]
    cases = manager.plotEvaluate(ser_X,
        title="State 1 evaluation for T2.0", is_plot=IS_PLOT)


if __name__ == '__main__':
  unittest.main()


