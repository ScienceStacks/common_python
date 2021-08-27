from common_python.classifier.case_query  \
    import CaseQuery, CaseCollectionQuery
from common_python.classifier.feature_set import FeatureVector
from common_python.classifier.case_manager import CaseManager, Case
from common_python import constants as cn
from common.trinary_data import TrinaryData
from common_python.tests.classifier import helpers
from common_python.util.persister import Persister

import copy
import matplotlib.pyplot as plt
import os
import pandas as pd
import unittest

IGNORE_TEST = True
IS_PLOT = True
FEATURE = "Rv2009"
FEATURE_VECTOR_MINUS_1 = FeatureVector({FEATURE: -1})
FEATURE_VECTOR_ZERO = FeatureVector({FEATURE: 0})
FEATURE_VECTOR_PLUS_1 = FeatureVector({FEATURE: 1})
CLASS = 1
DATA = TrinaryData(is_regulator=True, is_averaged=False, is_dropT1=False)
DF_X = DATA.df_X
SER_Y_ALL = DATA.ser_y
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PERSISTER_PATH = os.path.join(TEST_DIR, "test_case_query.pcl")
PERSISTER = Persister(PERSISTER_PATH)
if PERSISTER.isExist():
  CASE_COLS = PERSISTER.get()
else:
  case_managers = CaseManager.mkCaseManagers(DF_X, SER_Y_ALL,
      n_estimators=300)
  [c.build() for c in case_managers.values()]
  CASE_COLS = [c.case_col for c in list(case_managers.values())]
  PERSISTER.set(CASE_COLS)
IDX1 = 5
CASE_COL_DCT = {n: CASE_COLS[n] for n in range(len(CASE_COLS))}

def getSerDesc():
  df_desc = helpers.PROVIDER.df_go_terms.copy()
  df_desc = df_desc.set_index("GENE_ID")
  return df_desc["GO_Term"]


class TestCaseQuery(unittest.TestCase):

  def setUp(self):
    self.case_col = copy.deepcopy(CASE_COLS[CLASS])
    self.query = CaseQuery(self.case_col)
    self.keys = list(self.case_col.keys())
    dct = {k: self.case_col[k] for k in self.keys[0:IDX1]}
    self.query_short = CaseQuery(dct)
    dct = {k: self.case_col[k] for k in self.keys[IDX1:]}
    self.query_long = CaseQuery(dct)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertTrue(self.query.case_col == self.case_col)
 
  def testUnion(self):
    if IGNORE_TEST:
      return
    case_col = CaseQuery.union(self.query.case_col, 
         other_query=self.query_long)
    self.assertEqual(len(case_col), len(self.query))
    case_col = CaseQuery.union(self.query_short.case_col,
        other_query=self.query_short)
    self.assertEqual(len(case_col), IDX1)
 
  def testIntersection(self):
    if IGNORE_TEST:
      return
    case_col = CaseQuery.intersection(self.query.case_col,
       other_query=self.query_short)
    self.assertTrue(case_col == self.query_short.case_col)
    case_col = CaseQuery.intersection(self.query_long.case_col,
        other_query=self.query_short)
    self.assertEqual(len(case_col), 0)
 
  def testDifference(self):
    if IGNORE_TEST:
      return
    case_col = CaseQuery.difference(self.query.case_col,
        other_query=self.query_short)
    self.assertTrue(case_col == self.query_long.case_col)
    case_col = CaseQuery.difference(self.query.case_col,
        other_query=self.query)
    self.assertEqual(len(case_col), 0)

  def testSelectByDescription(self):
    if IGNORE_TEST:
      return
    num_case = len(self.query)
    #
    term = "cell"
    ser_desc = getSerDesc()
    case_col = CaseQuery.selectByDescription(
        self.query.case_col, ser_desc=ser_desc, terms=[term])
    self.assertLess(len(case_col), num_case)
    new_case_col = CaseQuery.selectByDescription(
        self.query.case_col, ser_desc=ser_desc, terms=[term])
    self.assertEqual(len(new_case_col), len(case_col))
    new_case_col = CaseQuery.selectByDescription(
        self.query.case_col, ser_desc=ser_desc, terms=["hypoxia"])
    self.assertLess(len(new_case_col), len(case_col))

  def testSelectByFeatureVector(self):
    if IGNORE_TEST:
      return
    case_col = CaseQuery.selectByFeatureVector(
        self.query.case_col, feature_vector=FEATURE_VECTOR_ZERO)
    self.assertLess(len(case_col), len(self.query))
    for key in case_col.keys():
      self.assertTrue(FEATURE in key)

  def testSelect(self):
    if IGNORE_TEST:
      return
    query = CaseQuery.select(CaseQuery.union, self.query_short,
        other_query=self.query_long)
    self.assertEqual(len(query), len(self.query))


class TestCaseCollectionQuery(unittest.TestCase):

  def setUp(self):
    self.case_col_dct = copy.deepcopy(CASE_COL_DCT)
    self.collection_query = CaseCollectionQuery(self.case_col_dct)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(len(self.collection_query.names),
        len(self.collection_query.case_col_dct))

  def testToDataframe(self):
    if IGNORE_TEST:
      return
    df = self.collection_query.toDataframe()
    self.assertTrue(isinstance(df, pd.DataFrame))
    num_case = sum([len(c) for c in
        self.collection_query.case_query_dct.values()])
    self.assertLessEqual(len(df), num_case)

  def testSelect(self):
    # TESTING
    ser_desc = getSerDesc()
    collection_query = self.collection_query.select(
        CaseQuery.selectByDescription,
        ser_desc=ser_desc, terms=["hypoxia"])
    import pdb; pdb.set_trace()
    
  def testPlotHeatmap(self):
    if IGNORE_TEST:
      return
    terms = ["fatty acid", "hypoxia"]
    df_desc = helpers.PROVIDER.df_go_terms.copy()
    df_desc = df_desc.set_index("GENE_ID")
    ser_desc = df_desc["GO_Term"]
    for manager in self.collection.manager_dct.values():
      manager.filterCaseByDescription(ser_desc, include_terms=terms)
    df = self.collection.plotHeatmap(is_plot=IS_PLOT)
    df.to_csv("t.csv")
    self.assertTrue(isinstance(df, pd.DataFrame))

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
