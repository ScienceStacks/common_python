import common_python.constants as cn
from common_python.testing import helpers
from common_python.classifier import feature_analyzer
from common_python.classifier.feature_analyzer import FeatureAnalyzer
from common_python.tests.classifier import helpers as test_helpers

import copy
import os
import pandas as pd
import numpy as np
from sklearn import svm
import shutil
import time
import unittest

IGNORE_TEST = False
IS_SCALE = False  # Do scale tests
IS_REPORT = False
IS_PLOT = False
CLASS = 1
DF_X, SER_Y_ALL = test_helpers.getDataLong()
CLASSES = list(SER_Y_ALL.unique())
# Make binary classes (for CLASS)
SER_Y = pd.Series([
    cn.PCLASS if v == CLASS else cn.NCLASS
    for v in SER_Y_ALL], index=SER_Y_ALL.index)
NUM_CROSS_ITER = 5
NUM_CROSS_ITER_ACCURATE = 50
CLF = svm.LinearSVC()
FEATURE1 = "Rv0158"
FEATURE2 = "Rv1460"
FEATURES = [FEATURE1, FEATURE2]
# Number of features used for scaling runs
NUM_FEATURE_SCALE = 100
# File paths for tests
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PERSISTER_PATH = os.path.join(TEST_DIR,
    "persister.pcl")
TMP_DIR_DCT = {CLASS:
     os.path.join(TEST_DIR, "tmp_feature_analyzer_%d") % CLASS}
TEST_SER_PATH = os.path.join(TEST_DIR, "ser.csv")
# Existing data
TEST_DIR_PATH = os.path.join(TEST_DIR,
                           "test_feature_analyzer_%d" % CLASS)
TEST_DIR_PATH_DCT = {CLASS: TEST_DIR_PATH}
ANALYZER = test_helpers.getFeatureAnalyzer()
ANALYZER_DCT = {test_helpers.CLASS: ANALYZER}

class TestFeatureAnalyzer(unittest.TestCase):

  def _init(self):
    self.df_X = copy.deepcopy(DF_X[FEATURES])
    self.ser_y = copy.deepcopy(SER_Y)
    self.clf = copy.deepcopy(CLF)
    self.analyzer = feature_analyzer.FeatureAnalyzer(
        self.clf, self.df_X, self.ser_y,
        is_report=IS_REPORT,
        num_cross_iter=NUM_CROSS_ITER_ACCURATE)
    self.analyzer_dct = ANALYZER_DCT

  def _remove(self):
    paths = [TEST_SER_PATH, TEST_PERSISTER_PATH]
    paths.extend(list(TMP_DIR_DCT.values()))
    for path in paths:
      if os.path.isdir(path):
        shutil.rmtree(path)
      if os.path.isfile(path):
        os.remove(path)

  def setUp(self):
    if IGNORE_TEST:
      return
    self._remove()
    self._init()

  def tearDown(self):
    self._remove()

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(len(self.analyzer._partitions),
        NUM_CROSS_ITER_ACCURATE)

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
    _ = analyzer.ser_sfa
    self._report("test_ser_sfa_scale", start)

  def test_ser_sfa(self):
    if IGNORE_TEST:
      return
    ser = self.analyzer.ser_sfa
    trues = [isinstance(v, float) for v in ser.values]
    self.assertTrue(all(trues))

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
    _ = analyzer.df_cpc
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
    _ = analyzer.df_ipa
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

  def testPlotSFA(self):
    if IGNORE_TEST:
      return
    self._init()
    analyzers = list(self.analyzer_dct.values()) * 6
    feature_analyzer.plotSFA(analyzers, is_plot=IS_PLOT)

  def testPlotCPC(self):
    if IGNORE_TEST:
      return
    self.analyzer_dct[CLASS].plotCPC(is_plot=IS_PLOT)
    criteria = lambda v: v < 0.5
    self.analyzer_dct[CLASS].plotCPC(is_plot=IS_PLOT,
                                     criteria=criteria)

  def testPlotIPA(self):
    if IGNORE_TEST:
      return
    self.analyzer_dct[CLASS].plotIPA(is_plot=IS_PLOT)

  def testGetPath(self):
    dir_path = TMP_DIR_DCT[CLASS]
    path = FeatureAnalyzer._getPath(dir_path,
                                    feature_analyzer.CPC)
    self.assertTrue("csv" in path)

  def testMakeSer(self):
    if IGNORE_TEST:
      return
    index = ['a', 'b', 'c']
    ser = pd.Series(range(len(index)), index=index)
    ser.to_csv(TEST_SER_PATH)
    df = pd.read_csv(TEST_SER_PATH)
    ser_new = FeatureAnalyzer._makeSer(df, is_sort=False)
    self.assertTrue(ser.equals(ser_new))

  def _equals(self, analyzer1, analyzer2):
    for metric in feature_analyzer.METRICS:
      value1 = analyzer1.getMetric(metric)
      value2 = analyzer2.getMetric(metric)
      self.assertTrue(all(value1.eq(value2)))

  def testSerializeAndDeserialize(self):
    if IGNORE_TEST:
      return
    dir_path = TMP_DIR_DCT[CLASS]
    self.analyzer.serialize(dir_path,
        persister_path=TEST_PERSISTER_PATH)
    for name in feature_analyzer.VARIABLES:
      path = FeatureAnalyzer._getPath(dir_path, name)
      self.assertTrue(os.path.isfile(path))
    #
    analyzer = feature_analyzer.FeatureAnalyzer.deserialize(
      dir_path)
    self._equals(self.analyzer, analyzer)
    for metric in feature_analyzer.METRICS:
      m_old = self.analyzer.getMetric(metric)
      m_new = analyzer.getMetric(metric)
      self.assertTrue(all(m_old.eq(m_new)))

  def testSerializeWithPersister(self):
    if IGNORE_TEST:
      return
    # Serialize the existing data
    dir_path = TMP_DIR_DCT[CLASS]
    self.analyzer.serialize(dir_path,
        persister_path=TEST_PERSISTER_PATH)
    # Create a new analyzer with no data
    analyzer = FeatureAnalyzer(None,
        pd.DataFrame(), pd.Series())
    new_analyzer = analyzer.serialize(dir_path,
        is_restart=False,
        persister_path=TEST_PERSISTER_PATH)
    self._equals(self.analyzer, new_analyzer)

  def testmakeAnalyzers(self):
    if IGNORE_TEST:
      return
    dct = feature_analyzer.deserialize(TEST_DIR_PATH_DCT)
    for cl in dct.keys():
      self.assertTrue(isinstance(dct[cl],
          feature_analyzer.FeatureAnalyzer))

  def testBackEliminate(self):
    if IGNORE_TEST:
      return
    self._init()
    features = [FEATURE1, FEATURE2]
    score = self.analyzer.score(features)
    be_result = self.analyzer.backEliminate(features)
    self.assertTrue(np.isclose(score, be_result.score))
    


if __name__ == '__main__':
  unittest.main()
