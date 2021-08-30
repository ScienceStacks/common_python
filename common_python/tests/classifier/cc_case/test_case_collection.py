import common_python.constants as cn
from common_python.classifier.feature_set  \
    import FeatureSet, FeatureVector
from common_python.classifier.cc_case.case_core \
    import  Case, FeatureVectorStatistic
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

IGNORE_TEST = False
IS_PLOT = False
FEATURE = "Rv2009"
FEATURE_VECTOR_MINUS_1 = FeatureVector({FEATURE: -1})
FEATURE_VECTOR_ZERO = FeatureVector({FEATURE: 0})
FEATURE_VECTOR_PLUS_1 = FeatureVector({FEATURE: 1})
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTION_PATH = os.path.join(TEST_DIR, "case_collection.csv")
DF_X_PATH = os.path.join(TEST_DIR, "feature_values.csv")
SER_Y_PATH = os.path.join(TEST_DIR, "class_values.csv")
TMP_FILE = os.path.join(TEST_DIR, "test_case_collection_tmp.csv")
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
SER_DESC = helpers.getDescription()
#
PERSISTER_FILE = "test_case_collection.pcl"
persister = Persister(PERSISTER_FILE)
if persister.isExist():
  CASE_COLLECTION = persister.get()
else:
  CASE_COLLECTION = CaseCollection.deserialize(COLLECTION_PATH,
      df_X_path=DF_X_PATH, ser_y_path=SER_Y_PATH)
  persister.set(CASE_COLLECTION)
TMP_FILES = [TMP_FILE]


class TestCaseCollection(unittest.TestCase):

  def setUp(self):
    self.collection = copy.deepcopy(CASE_COLLECTION)
    self.keys = list(self.collection.keys())
    self._remove()

  def tearDown(self):
    self._remove()

  def _remove(self):
    for ffile in TMP_FILES:
      if os.path.isfile(ffile):
        os.remove(ffile)

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
    for col in [cn.SIGLVL, cn.PRIOR_PROB, cn.NUM_SAMPLE, cn.NUM_POS,
        cn.INSTANCE_STR]:
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
    collection = self.collection.difference(CaseCollection({},
        df_X=DF_X, ser_y=SER_Y))
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

  def testFindByFeatureVector(self):
    if IGNORE_TEST:
      return
    case_col = self.collection.findByFeatureVector(
        feature_vector=FEATURE_VECTOR_ZERO)
    self.assertLess(len(case_col), len(self.collection))
    for key in case_col.keys():
      self.assertTrue(FEATURE in key)

  def testFindIsContained(self):
    if IGNORE_TEST:
      return
    ser_X = DF_X.T["T2.0"]
    feature_vector = FeatureVector.make(ser_X)
    case_col = self.collection.findIsContained(
        feature_vector=feature_vector)
    self.assertGreater(len(case_col), 100)
    self.assertLess(len(case_col), len(self.collection))
    for case in case_col.values():
      for feature, value in case.feature_vector.dict.items():
        self.assertEqual(ser_X.loc[feature], value)

  def testMake(self):
    if IGNORE_TEST:
      return
    collection = self.collection.make(self.collection.values())
    self.assertTrue(collection == self.collection)

  def testSerialize(self):
    if IGNORE_TEST:
      return
    self.collection.serialize(TMP_FILE)
    collection = CaseCollection.deserialize(TMP_FILE,
        df_X_path=DF_X_PATH, ser_y_path=SER_Y_PATH)
    self.assertTrue(collection == self.collection)

  def testDeserialize(self):
    if IGNORE_TEST:
      return
    collection = CaseCollection.deserialize(collection_path=COLLECTION_PATH,
        df_X_path=DF_X_PATH, ser_y_path=SER_Y_PATH)
    self.assertTrue(collection == self.collection)

  def testPlotEvaluate(self):
    if IGNORE_TEST:
      return
    ser_X = DF_X.loc["T14.0", :]
    new_collection = self.collection.findByDescription(
        ser_desc=SER_DESC, terms=["fatty acid"])
    collection = new_collection.plotEvaluate(ser_X,
        title="State 1 evaluation for T14.0", is_plot=IS_PLOT)
    ser_X = DF_X.loc["T2.0", :]
    collection = new_collection.plotEvaluate(ser_X,
        title="State 1 evaluation for T2.0", is_plot=IS_PLOT)

  def testCountCases(self):
    if IGNORE_TEST:
      return
    frac_pos_all, num_total_all = self.collection.countCases(
        is_drop_duplicate=False)
    frac_pos, num_total = self.collection.countCases()
    self.assertGreater(num_total_all, num_total)
    self.assertGreater(frac_pos, frac_pos_all)


if __name__ == '__main__':
  unittest.main()
