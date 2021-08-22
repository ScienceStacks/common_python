from common_python.classifier.case_manager_collection  \
    import CaseManagerCollection
from common_python import constants as cn
from common.trinary_data import TrinaryData
from common_python.tests.classifier import helpers
from common_python.util.persister import Persister

import copy
import matplotlib.pyplot as plt
import os
import pandas as pd
import unittest

IGNORE_TEST = False
IS_PLOT = False
CLASS = 1
DATA = TrinaryData(is_regulator=True, is_averaged=False, is_dropT1=False)
DF_X = DATA.df_X
SER_Y_ALL = DATA.ser_y
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PERSISTER_PATH = os.path.join(TEST_DIR, "test_case_manager_collection.pcl")
PERSISTER = Persister(PERSISTER_PATH)
if PERSISTER.isExist():
  CASE_MANAGER_COLLECTION = PERSISTER.get()
else:
  CASE_MANAGER_COLLECTION = CaseManagerCollection(DF_X, SER_Y_ALL,
      n_estimators=300)
  CASE_MANAGER_COLLECTION.build()
  PERSISTER.set(CASE_MANAGER_COLLECTION)


class TestCaseManagerCollection(unittest.TestCase):

  def setUp(self):
    self.collection = copy.deepcopy(CASE_MANAGER_COLLECTION)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(len(self.collection.manager_dct),
        len(set(SER_Y_ALL.values)))

  def testBuild(self):
    if IGNORE_TEST:
      return
    self.collection.build()
    for manager in self.collection.manager_dct.values():
      self.assertTrue(isinstance(manager.case_dct, dict))

  def testPlotHeatmap(self):
    if IGNORE_TEST:
      return
    self.collection.build()
    terms = ["fatty acid", "hypoxia"]
    df_desc = helpers.PROVIDER.df_go_terms.copy()
    df_desc = df_desc.set_index("GENE_ID")
    ser_desc = df_desc["GO_Term"]
    for manager in self.collection.manager_dct.values():
      manager.filterCaseByDescription(ser_desc, include_terms=terms)
    df = self.collection.plotHeatmap(is_plot=True)
    self.assertTrue(isinstance(df, pd.DataFrame))


if __name__ == '__main__':
  unittest.main()
