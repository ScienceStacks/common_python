import common_python.constants as cn
from common_python.classifier.feature_set  \
    import FeatureSet, FeatureVector
from common_python.classifier.cc_case.case_core \
    import  Case, FeatureVectorStatistic
from common_python.classifier.cc_case.case_builder import CaseBuilder
from common_python.classifier.cc_case.case_collection import CaseCollection
from common_python import constants as cn
from common_python.tests.classifier.cc_case import helpers
from common_python.util.persister import Persister

import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import unittest

IGNORE_TEST = True
IS_PLOT = True
FEATURE = "Rv2009"
FEATURE_VECTOR_MINUS_1 = FeatureVector({FEATURE: -1})
FEATURE_VECTOR_ZERO = FeatureVector({FEATURE: 0})
FEATURE_VECTOR_PLUS_1 = FeatureVector({FEATURE: 1})
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
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
#
PERSISTER_PATH = os.path.join(TEST_DIR, "test_case_collection.pcl")
PERSISTER = Persister(PERSISTER_PATH)
done = False
if PERSISTER.isExist():
  CASE_COLLECTION = PERSISTER.get()
else:
  case_builder  = CaseBuilder(DF_X, SER_Y)
  case_builder.build()
  CASE_COLLECTION = case_builder.case_col
  PERSISTER.set(CASE_COLLECTION)
#
SER_DESC = helpers.getDescription()


class TestCaseCollection(unittest.TestCase):

  def setUp(self):
    self.collection = copy.deepcopy(CASE_COLLECTION)
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

  def testSymmetricDifference(self):
    if IGNORE_TEST:
      return
    collection = self.collection.difference(self.collection)
    self.assertEqual(len(collection), 0)
    #
    def test(end_idx=0, start_idx=10, start1_trim=None,
        start2_trim=None, expected_length=0):
      collection1 = copy.deepcopy(self.collection)
      collection2 = copy.deepcopy(self.collection)
      for key in self.keys[0:end_idx]:
        del collection1[key]
      for key in self.keys[start_idx:]:
        del collection2[key]
      if start1_trim is not None:
        for key in self.keys[start1_trim:]:
          del collection1[key]
      if start2_trim is not None:
        for key in self.keys[start2_trim:]:
          del collection2[key]
      collection = collection1.symmetricDifference(collection2)
      self.assertEqual(len(collection), expected_length)
    #
    test(end_idx=0, start_idx=0, expected_length=len(self.collection))
    #
    idx = 10
    expected_length = len(self.collection)
    test(end_idx=idx, start_idx=idx, expected_length=expected_length)
    #
    offset = 4
    expected_length = len(self.collection) - offset
    test(end_idx=idx, start_idx=idx-offset, expected_length=expected_length)
    #
    start2_trim = len(self.collection) - idx
    expected_length = 2*idx
    test(end_idx=idx, start_idx=len(self.collection), start2_trim=start2_trim,
        expected_length=expected_length)

  def testFindtByDescription(self):
    if IGNORE_TEST:
      return
    num_case = len(self.collection)
    #
    term = "cell"
    case_col = self.collection.findByDescription(
        ser_desc=SER_DESC, terms=[term])
    self.assertLess(len(case_col), num_case)
    new_case_col = case_col.findByDescription(
        ser_desc=SER_DESC, terms=[term])
    self.assertTrue(case_col == new_case_col)
    new_case_col = self.collection.findByDescription(
        ser_desc=SER_DESC, terms=["hypoxia"])
    self.assertLess(len(new_case_col), len(case_col))

  def testSelectByFeatureVector(self):
    # TESTING
    case_col = self.collection.findByFeatureVector(
        feature_vector=FEATURE_VECTOR_ZERO)
    self.assertLess(len(case_col), len(self.collection))
    for key in case_col.keys():
      self.assertTrue(FEATURE in key)

  def testMake(self):
    if IGNORE_TEST:
      return
    collection = self.collection.make(self.collection.values())
    self.assertTrue(collection == self.collection)


if __name__ == '__main__':
  unittest.main()
