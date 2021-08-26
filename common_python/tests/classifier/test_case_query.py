from common_python.classifier.case_query  \
    import CaseQuery, CaseCollectionQuery
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
CLASS = 1
DATA = TrinaryData(is_regulator=True, is_averaged=False, is_dropT1=False)
DF_X = DATA.df_X
SER_Y_ALL = DATA.ser_y
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PERSISTER_PATH = os.path.join(TEST_DIR, "test_case_query.pcl")
PERSISTER = Persister(PERSISTER_PATH)
if PERSISTER.isExist():
  CASE_DCTS = PERSISTER.get()
else:
  CASE_MANAGER_COLLECTION = CaseManager.mkCaseManagers(DF_X, SER_Y_ALL,
      n_estimators=300)
  CASE_MANAGER_COLLECTION.build()
  PERSISTER.set(CASE_MANAGER_COLLECTION)


class TestCaseQuery(unittest.TestCase):

  def setUp(self):
    case_class_dct = copy.deepcopy(CASE_MANAGER_COLLECTION.manager_dct)
    keys = CASE_MANAGER_COLLECTION.manager_dct.keys()
    case_class_dct = {k:  [c for c in 
        CASE_MANAGER_COLLECTION.manager_dct[k].case_col.values()]
        for k in keys}
    self.query = CaseQuery(case_class_dct)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    key = self.query.classes[0]
    values = [v for v in self.query.case_class_dct[key]]
    self.assertTrue(isinstance(values[0], Case))

  def testToDataframe(self):
    if IGNORE_TEST:
      return
    df = self.query.toDataframe()
    size_dct = {c: len(df[c].dropna()) for c in df.columns}
    self.assertLess(size_dct[0], size_dct[1])
    self.assertLess(size_dct[4], size_dct[1])

  def testPlotHeatmap(self):
    # TESTING
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

  def testSelectByDescription(self):
    if IGNORE_TEST:
      return
    num_case = len(self.manager.case_col)
    #
    term = "cell"
    df_desc = helpers.PROVIDER.df_go_terms.copy()
    df_desc = df_desc.set_index("GENE_ID")
    ser_desc = df_desc["GO_Term"]
    cases = self.manager.selectCaseByDescription(ser_desc, terms=[term])
    self.assertLess(len(cases), num_case)
    new_cases = self.manager.selectCaseByDescription(ser_desc, cases=cases,
        terms=[term])
    self.assertEqual(len(new_cases), len(cases))
    new_cases = self.manager.selectCaseByDescription(ser_desc, cases=cases,
        terms=["hypoxia"])
    self.assertLess(len(new_cases), len(cases))
    import pdb; pdb.set_trace()



if __name__ == '__main__':
  unittest.main()
