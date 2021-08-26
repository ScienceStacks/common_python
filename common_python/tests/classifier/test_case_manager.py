import common_python.constants as cn
from common_python.classifier.feature_set  \
    import FeatureSet, FeatureVector
from common_python.classifier.case_manager  \
    import CaseManager, Case, FeatureVectorStatistic, CaseCollection
from common_python import constants as cn
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
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PERSISTER_PATH = os.path.join(TEST_DIR, "test_case_manager.pcl")
PERSISTER = Persister(PERSISTER_PATH)
if PERSISTER.isExist():
  CASE_MANAGER = PERSISTER.get()
else:
  CASE_MANAGER  = CaseManager(DF_X, SER_Y)
  CASE_MANAGER.build()
  PERSISTER.set(CASE_MANAGER)


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
    fv_statistic = FeatureVectorStatistic(COUNT, int(FRAC*COUNT), 0.5, SIGLVL)
    self.case = Case(FEATURE_VECTOR, fv_statistic)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(self.case.fv_statistic.num_sample, COUNT)

  def testEq(self):
    if IGNORE_TEST:
      return
    self.assertTrue(self.case == self.case)
    fv_statistic = FeatureVectorStatistic(COUNT - 1, int(FRAC*COUNT), 0.5,
        SIGLVL)
    case = Case(FEATURE_VECTOR, fv_statistic)
    self.assertFalse(case == self.case)


class TestCaseCollection(unittest.TestCase):

  def setUp(self):
    self.collection = CaseCollection(CASE_MANAGER.case_col)
    self.keys = list(self.collection.keys())

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertTrue(isinstance(self.collection, CaseCollection))

  def testSort(self):
    if IGNORE_TEST:
      return
    collection = copy.deepcopy(self.collection)
    collection.sort()
    num_item = len(collection)
    keys = list(collection.keys())
    for idx in np.random.randint(0, num_item - 1, 100):
      if idx < num_item:
        self.assertLess(keys[idx], keys[idx+1])
    self.assertTrue(self.collection == collection)

  def testEq(self):
    if IGNORE_TEST:
      return
    self.assertTrue(self.collection == self.collection)
    collection = copy.deepcopy(self.collection)
    del collection[self.keys[0]]
    self.assertFalse(collection == self.collection)

  def testToDataframe(self):
    if IGNORE_TEST:
      return
    df = self.collection.toDataframe()
    self.assertTrue(isinstance(df, pd.DataFrame))
    for col in [cn.SIGLVL, cn.PRIOR_PROB, cn.NUM_SAMPLE, cn.NUM_POS]:
      self.assertTrue(col in df.columns)
    self.assertGreater(len(df), 0)

  def testUnion(self):
    if IGNORE_TEST:
      return
    collection = self.collection.union(self.collection)
    self.assertTrue(collection == self.collection)
    #
    def test(end_idx=10, start_idx=10, expected_result=True):
      collection1 = copy.deepcopy(self.collection)
      collection2 = copy.deepcopy(self.collection)
      for key in self.keys[0:end_idx]:
        del collection1[key]
      for key in self.keys[start_idx:]:
        del collection2[key]
      collection = collection1.union(collection2)
      result = collection == self.collection
      self.assertEqual(result, expected_result)
    #
    test()
    test(end_idx=10, start_idx=14)
    test(end_idx=14, start_idx=10, expected_result=False)

  def testIntersection(self):
    if IGNORE_TEST:
      return
    collection = self.collection.intersection(self.collection)
    self.assertTrue(collection == self.collection)
    #
    def test(end_idx=10, start_idx=10, expected_length=0):
      collection1 = copy.deepcopy(self.collection)
      collection2 = copy.deepcopy(self.collection)
      for key in self.keys[0:end_idx]:
        del collection1[key]
      for key in self.keys[start_idx:]:
        del collection2[key]
      collection = collection1.intersection(collection2)
      self.assertEqual(len(collection), expected_length)
    #
    test()
    test(end_idx=10, start_idx=14, expected_length=4)
    test(end_idx=14, start_idx=10)

  def testDifference(self):
    if IGNORE_TEST:
      return
    collection = self.collection.difference(self.collection)
    self.assertEqual(len(collection), 0)
    #
    def test(end_idx=10, start_idx=10, expected_length=0):
      collection1 = copy.deepcopy(self.collection)
      collection2 = copy.deepcopy(self.collection)
      for key in self.keys[0:end_idx]:
        del collection1[key]
      for key in self.keys[start_idx:]:
        del collection2[key]
      collection = collection1.difference(collection2)
      self.assertEqual(len(collection), expected_length)
    #
    idx = 10
    expected_length = len(self.collection) - idx
    test(end_idx=idx, start_idx=idx, expected_length=expected_length)
    test(end_idx=10, start_idx=0, expected_length=len(self.collection)-10)
    #
    collection = self.collection.difference(CaseCollection({}))
    self.assertTrue(collection == self.collection)

  def testMake(self):
    if IGNORE_TEST:
      return
    collection = self.collection.make(self.collection.values())
    self.assertTrue(collection == self.collection)


class TestCaseManager(unittest.TestCase):

  def setUp(self):
    self.manager = copy.deepcopy(CASE_MANAGER)

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
    case_col = self.manager._getCases(dtree)
    self.assertTrue(len(case_col) > 3)
    if IS_PLOT:
      self.manager.displayCases(cases=case_col.values())
      plt.show()

  def testBuild(self):
    if IGNORE_TEST:
      return
    num_tree = 20
    manager = CaseManager(DF_X, SER_Y, n_estimators=num_tree)
    manager.build()
    self.assertGreater(len(manager.case_col), 2*num_tree)

  def testMkCaseManagers(self):
    if IGNORE_TEST:
      return
    num_tree = 10
    manager_dct = CaseManager.mkCaseManagers(DF_X, SER_Y_ALL,
        n_estimators=num_tree)
    classes = set(SER_Y_ALL.values)
    self.assertEqual(len(manager_dct), len(classes))


if __name__ == '__main__':
  unittest.main()
