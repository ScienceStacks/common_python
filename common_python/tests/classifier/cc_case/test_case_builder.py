import common_python.constants as cn
from common_python.classifier.feature_set  \
    import FeatureSet, FeatureVector
from common_python.classifier.cc_case.case_builder import CaseBuilder
from common_python.classifier.cc_case.case_core import  \
    Case, CaseCollection, FeatureVectorStatistic
from common.trinary_data import TrinaryData
from common_python.tests.classifier import helpers
from common_python.util.persister import Persister

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
DATA = TrinaryData(is_regulator=True, is_averaged=False, is_dropT1=False)
DF_X = DATA.df_X
SER_Y_ALL = DATA.ser_y
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
#
CASE_BUILDER_VERSION = 2
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PERSISTER_PATH = os.path.join(TEST_DIR, "test_case_builder.pcl")
PERSISTER = Persister(PERSISTER_PATH)
done = False
if PERSISTER.isExist():
  CASE_BUILDER = PERSISTER.get()
  if "version" in dir(CASE_BUILDER):
    if CASE_BUILDER.version == CASE_BUILDER_VERSION:
      done = True
if not done:
  CASE_BUILDER  = CaseBuilder(DF_X, SER_Y)
  CASE_BUILDER.build()
  PERSISTER.set(CASE_BUILDER)


class TestCaseBuilder(unittest.TestCase):

  def setUp(self):
    self.builder = copy.deepcopy(CASE_BUILDER)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertGreater(len(self.builder._features), 0)

  def testGetCompatibleFeatureValues(self):
    if IGNORE_TEST:
      return
    values = self.builder._getCompatibleFeatureValues(FEATURE_B, 1)
    self.assertEqual(values, [-1, 0, 1])
    values = self.builder._getCompatibleFeatureValues(FEATURE_B, 0.5)
    self.assertEqual(values, [-1, 0])

  def testGetStatistic(self):
    if IGNORE_TEST:
      return
    statistic = self.builder._getFeatureVectorStatistic(FEATURE_VECTOR)
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
    case_col = self.builder._getCases(dtree)
    self.assertTrue(len(case_col) > 3)
    if IS_PLOT:
      self.builder.displayCases(cases=case_col.values())
      plt.show()

  def testBuild(self):
    if IGNORE_TEST:
      return
    num_tree = 20
    builder = CaseBuilder(DF_X, SER_Y, n_estimators=num_tree)
    builder.build()
    self.assertGreater(len(builder.case_col), 2*num_tree)

  def testMake(self):
    if IGNORE_TEST:
      return
    num_tree = 10
    multi = CaseBuilder.make(DF_X, SER_Y_ALL,
        n_estimators=num_tree)
    classes = set(SER_Y_ALL.values)
    self.assertEqual(len(multi.collection_dct), len(classes))


if __name__ == '__main__':
  unittest.main()
