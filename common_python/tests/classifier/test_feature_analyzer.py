import common_python.constants as cn
from common_python.testing import helpers
from common_python.classifier import feature_analyzer
from common_python.tests.classifier import helpers as test_helpers
from common_python.testing import helpers

import copy
import os
import pandas as pd
import numpy as np
from sklearn import svm
import time
import unittest

IGNORE_TEST = False
IS_SCALE = False  # Do scale tests
IS_REPORT = False
CLASS = 1
DF_X, SER_Y = test_helpers.getDataLong()
# Make binary classes (for CLASS)
SER_Y = pd.Series([
    cn.PCLASS if v == CLASS else cn.NCLASS
    for v in SER_Y], index=SER_Y.index)
NUM_CROSS_ITER = 5
NUM_CROSS_ITER_ACCURATE = 50
CLF = svm.LinearSVC()
FEATURE1 = "Rv0158"
FEATURE2 = "Rv1460"
FEATURES = [FEATURE1, FEATURE2]
# Number of features used for scaling runs
NUM_FEATURE_SCALE = 100


class TestFeatureAnalyzer(unittest.TestCase):
  
  def setUp(self):
    self.df_X = copy.deepcopy(DF_X[FEATURES])
    self.ser_y = copy.deepcopy(SER_Y)
    self.clf = copy.deepcopy(CLF)
    self.analyzer = feature_analyzer.FeatureAnalyzer(
        self.clf, self.df_X, self.ser_y,
        is_report=IS_REPORT,
        num_cross_iter=NUM_CROSS_ITER_ACCURATE)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(len(self.analyzer._partitions),
        NUM_CROSS_ITER_ACCURATE)

  def test_ser_sfa(self):
    if IGNORE_TEST:
      return
    ser = self.analyzer.ser_sfa
    trues = [isinstance(v, float) for v in ser.values]
    self.assertTrue(all(trues))

  def _report(self, method, start):
    print("\n*** Ran %s in %4.2f seconds" %
        (method, time.time() - start))

  def test_ser_sfa_scale(self):
    if IGNORE_TEST:
      return
    if not IS_SCALE:
      return
    df_X = self._makeDFX(NUM_FEATURE_SCALE)
    start = time.time()
    analyzer = feature_analyzer.FeatureAnalyzer(
        self.clf, df_X, self.ser_y,
        num_cross_iter=NUM_CROSS_ITER)
    ser = analyzer.ser_sfa
    self._report("test_ser_sfa_scale", start)

  def _makeDFX(self, num_cols):
    features = list(DF_X.columns.tolist())
    features = features[:num_cols]
    return DF_X[features].copy()

  def test_df_cpc(self):
    if IGNORE_TEST:
      return
    df = self.analyzer.df_cpc
    self.assertEqual(df.loc[FEATURE1, FEATURE1], 0)
    self.assertTrue(np.isclose(
        df.loc[FEATURE2, FEATURE2], 1))
    self.assertTrue(helpers.isValidDataFrame(df,
      [FEATURE1, FEATURE2]))

  def test_df_cpc_scale(self):
    if IGNORE_TEST:
      return
    if not IS_SCALE:
      return
    num_cols = int(np.sqrt(NUM_FEATURE_SCALE))
    df_X = self._makeDFX(num_cols)
    analyzer = feature_analyzer.FeatureAnalyzer(
        self.clf, df_X, self.ser_y,
        num_cross_iter=NUM_CROSS_ITER)
    start = time.time()
    df = analyzer.df_cpc
    self._report("test_df_cpc_scale", start)

  def test_df_ipa(self):
    if IGNORE_TEST:
      return
    df = self.analyzer.df_ipa
    self.assertTrue(helpers.isValidDataFrame(df,
      [FEATURE1, FEATURE2]))
    trues = [isinstance(v, float) for v in 
        np.reshape(df.values, len(df)*len(df.columns))]
    self.assertTrue(all(trues))

  def test_df_ipa_scale(self):
    if IGNORE_TEST:
      return
    if not IS_SCALE:
      return
    num_cols = int(np.sqrt(NUM_FEATURE_SCALE))
    df_X = self._makeDFX(num_cols)
    analyzer = feature_analyzer.FeatureAnalyzer(
        self.clf, df_X, self.ser_y,
        num_cross_iter=NUM_CROSS_ITER)
    start = time.time()
    df = analyzer.df_ipa
    self._report("test_df_ipa_scale", start)

  def testReportProgress(self):
    if IGNORE_TEST:
      return
    self.analyzer._reportProgress(
       feature_analyzer.SFA, 0, 10)
    self.analyzer._reportProgress(
       feature_analyzer.SFA, 11, 10)
    #
    INTERVAL = 5
    analyzer = feature_analyzer.FeatureAnalyzer(
        self.clf, self.df_X, self.ser_y,
        is_report=IS_REPORT,
        report_interval = INTERVAL,
        num_cross_iter=NUM_CROSS_ITER_ACCURATE)
    analyzer._reportProgress(
       feature_analyzer.SFA, 0, 10)
    analyzer._reportProgress(
       feature_analyzer.SFA, INTERVAL + 1, 10)



if __name__ == '__main__':
  unittest.main()
