import common_python.constants as cn
from common_python.testing import helpers
from common_python.classifier import feature_analyzer
from common_python.classifier.feature_analyzer import FeatureAnalyzer
from common_python.tests.classifier import helpers as test_helpers
import common.constants as xcn

import copy
import os
import pandas as pd
import numpy as np
from sklearn import svm
import shutil
import time
import unittest

IGNORE_TEST = True
IS_SCALE = False  # Do scale tests
IS_REPORT = False
IS_PLOT = False
CLASS = 1
DF_X, SER_Y_ALL = test_helpers.getDataLong()
STATES = list(SER_Y_ALL.unique())
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
TEST_DIR_PAT = os.path.join(TEST_DIR, "feature_analyzer_%d")
TEST_SER_PATH = os.path.join(TEST_DIR, "ser.csv")
# Pattern for the serialization directories
SERIAL_PAT = os.path.join(xcn.DATA_DIR, "feature_analyzer")
SERIAL_PAT = os.path.join(SERIAL_PAT, "%d")


class TestFeatureAnalyzer(unittest.TestCase):

  def _init(self):
    self.df_X = copy.deepcopy(DF_X[FEATURES])
    self.ser_y = copy.deepcopy(SER_Y)
    self.clf = copy.deepcopy(CLF)
    self.analyzer = feature_analyzer.FeatureAnalyzer(
        self.clf, self.df_X, self.ser_y,
        is_report=IS_REPORT,
        num_cross_iter=NUM_CROSS_ITER_ACCURATE)
    if False:
      self.analyzer_dct = {s:
          feature_analyzer.FeatureAnalyzer.deserialize(SERIAL_PAT % s)
          for s in STATES}

  def _remove(self):
    paths = [TEST_SER_PATH]
    for state in STATES:
      path = TEST_DIR_PAT % state
      paths.append(path)
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
    _ = analyzer.ser_sfa
    self._report("test_ser_sfa_scale", start)

  # FIXME:  (2) wrong count of results
  def test_fset_fea_acc(self):
    if IGNORE_TEST:
      return
    self._init()
    analyzer = self.analyzer_dct[1]
    ser = analyzer.fset_fea_acc
    num_feature = len(analyzer._features)
    expected = num_feature*(num_feature-1)/2 + num_feature
    self.assertEqual(expected, len(ser))
    trues = [(v <= 1) and (v >= 0) for v in ser]
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
    feature_analyzer.plotSFA(self.analyzer_dct.values(),
                             is_plot=IS_PLOT)

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
    path = FeatureAnalyzer._getPath(TEST_DIR_PAT % CLASS,
                                    feature_analyzer.CPC)
    os.mkdir(path)
    self.assertTrue(os.path.isdir(path))
    os.rmdir(path)

  def testMakeSer(self):
    if IGNORE_TEST:
      return
    index = ['a', 'b', 'c']
    ser = pd.Series(range(len(index)), index=index)
    ser.to_csv(TEST_SER_PATH)
    df = pd.read_csv(TEST_SER_PATH)
    ser_new = FeatureAnalyzer._makeSer(df, is_sort=False)
    self.assertTrue(ser.equals(ser_new))

  def testSerialize(self):
    self._init()
    self.analyzer.serialize(TEST_DIR_PAT % CLASS)
    # TODO: add tests



if __name__ == '__main__':
  unittest.main()
