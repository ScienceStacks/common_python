from common_python.classifier.cc_case.case_multi_collection  \
    import CaseMultiCollection
from common_python.classifier.cc_case.case_collection import CaseCollection 
from common_python.classifier.feature_set import FeatureVector
from common_python.classifier.cc_case.case_builder import CaseBuilder
from common_python import constants as cn
from common_python.tests.classifier.cc_case import helpers
from common_python.util.persister import Persister

import copy
import matplotlib.pyplot as plt
import os
import pandas as pd
import unittest

IGNORE_TEST = False
IS_PLOT = False
DF_X, SER_Y = helpers.getData()
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SERIALIZED_PATH = os.path.join(TEST_DIR, "case_multi_collection.csv")
TMP_PATH = os.path.join(TEST_DIR, "test_case_multi_collection_tmp.csv")
df_X_path = os.path.join(TEST_DIR, "feature_values.csv")
ser_y_path = os.path.join(TEST_DIR, "class_values.csv")
PERSISTER_PATH = os.path.join(TEST_DIR, "test_case_multi_collection.pcl")
persister = Persister(PERSISTER_PATH)
if persister.isExist():
  CASE_MULTI_COLLECTION = persister.get()
else:
  CASE_MULTI_COLLECTION = CaseMultiCollection.deserialize(SERIALIZED_PATH,
      df_X_path=df_X_path, ser_y_path=ser_y_path)
  persister.set(CASE_MULTI_COLLECTION)
TMP_PATHS = [TMP_PATH]
SER_DESC = helpers.getDescription()


class TestCaseMultiCollection(unittest.TestCase):

  def setUp(self):
    self.multi = copy.deepcopy(CASE_MULTI_COLLECTION)
    self._removeFiles()

  def tearDown(self):
    self._removeFiles()

  def _removeFiles(self):
    for ffile in TMP_PATHS:
      if os.path.isfile(ffile):
        os.remove(ffile)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertTrue(isinstance(self.multi.collection_dct, dict))

  def testEq(self):
    if IGNORE_TEST:
      return
    self.assertTrue(self.multi == self.multi)
    multi = copy.deepcopy(self.multi)
    del multi.collection_dct[0]
    del multi.names[0]
    self.assertFalse(multi == self.multi)

  def testSerialize(self):
    if IGNORE_TEST:
      return
    self.multi.serialize(TMP_PATH)
    multi = CaseMultiCollection.deserialize(TMP_PATH)
    self.assertTrue(multi == self.multi)

  def testDeserialize(self):
    if IGNORE_TEST:
      return
    multi = CaseMultiCollection.deserialize(SERIALIZED_PATH,
        df_X_path=df_X_path, ser_y_path=ser_y_path)
    self.assertTrue(multi == self.multi)

  def testToDataframe(self):
    if IGNORE_TEST:
      return
    df1 = self.multi.toDataframe()
    self.assertTrue(isinstance(df1, pd.DataFrame))
    num_case = sum([len(c) for c in
        self.multi.collection_dct.values()])
    self.assertLessEqual(len(df1), num_case)
    #
    df2 = self.multi.toDataframe(max_sl=0.05)
    self.assertGreater(len(df2), len(df1))

  def testSelect(self):
    if IGNORE_TEST:
      return
    multi = self.multi.select(
        CaseCollection.selectByDescription,
        ser_desc=SER_DESC, terms=["hypoxia"])
    for name, case_col in self.multi.collection_dct.items():
      if name > 0:
        self.assertGreater(len(case_col), len(multi.collection_dct[name]))

  def testBinarySelect(self):
    if IGNORE_TEST:
      return
    multi = self.multi.select(
        CaseCollection.selectByDescription,
        ser_desc=SER_DESC, terms=["hypoxia"])
    new_multi = multi.binarySelect(CaseCollection.union, multi)
    self.assertTrue(new_multi == multi)
    #
    new_multi = self.multi.binarySelect(CaseCollection.difference, multi)
    newer_multi = new_multi.binarySelect(CaseCollection.union, multi)
    self.assertTrue(newer_multi == self.multi)

  def _filterCases(self):
    terms = ["fatty acid", "hypoxia"]
    return self.multi.select(CaseCollection.selectByDescription,
        ser_desc=SER_DESC, terms=terms)
   
  def testPlotHeatmap(self):
    if IGNORE_TEST:
      return
    multi = self._filterCases()
    df1 = multi.plotHeatmap(is_plot=IS_PLOT, title="All Cases")
    self.assertTrue(isinstance(df1, pd.DataFrame))
    #
    feature_vector = FeatureVector.make(DF_X.T["T2.0"])
    df2 = multi.plotHeatmap(feature_vector=feature_vector, is_plot=IS_PLOT,
        title="T2.0")
    self.assertLess(len(df2), len(df1))
   
  def testPlotBars(self):
    if IGNORE_TEST:
      return
    multi = self.multi
    multi = self._filterCases()
    multi.plotBars(is_plot=IS_PLOT, title="All Cases")
    #
    feature_vector = FeatureVector.make(DF_X.T["T2.0"])
    multi.plotBars(feature_vector=feature_vector, is_plot=IS_PLOT,
        title="T2.0", max_sl=1e-4)
   
  def testPlotBars2(self):
    if IGNORE_TEST:
      return
    ser_X = DF_X.loc["T14.0", :]
    multi = self._filterCases()
    multi.plotBars(FeatureVector(ser_X), is_plot=IS_PLOT)
   
  def testPlotBars3(self):
    if IGNORE_TEST:
      return
    ser_X = DF_X.loc["T14.0", :]
    self.multi.plotBars(FeatureVector(ser_X), is_plot=IS_PLOT, expected_class=3)
   
  def testPlotBarsForFeatures(self):
    if IGNORE_TEST:
      return
    df = DF_X[DF_X.index.str.contains(".0")]
    self.multi.plotBarsForFeatures(df, 5, 5, suptitle = "rep 0",
       ser_y=SER_Y, fontsize=8)


if __name__ == '__main__':
  unittest.main()
