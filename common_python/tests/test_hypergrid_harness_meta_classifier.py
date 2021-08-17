from common_python.classifier.hypergrid_harness_meta_classifier  \
    import HypergridHarnessMetaClassifier
from common_python.classifier \
    import hypergrid_harness_meta_classifier
from common_python.classifier.hypergrid_harness  \
    import Vector, Plane
from common_python.classifier.meta_classifier  \
    import MetaClassifierDefault, MetaClassifierPlurality,  \
    MetaClassifierAugment, MetaClassifierAverage, \
    MetaClassifierEnsemble
from common_python.testing import helpers
import common_python.constants as cn

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import unittest

IGNORE_TEST = False
IS_PLOT = False
NUM_DIM = 2
NUM_POINT = 25
NUM_REPL = 3
OFFSET = 0
IDX_PLURALITY = 0
IDX_DEFAULT = 1
IDX_AUGMENT = 2
IDX_AVERAGE = 3
IDX_ENSEMBLE = 4
IMPURITY = 0
SCORE_PLURALITY = 0.5
SCORE_IDEAL = 1.0
STD = 1.0
STDS = np.repeat(STD, NUM_DIM)
TEST_DATA_PTH = os.path.join(cn.TEST_DIR,
    "test_hypergrid_harness_meta_classifier.csv")
TEST_DATA_CREATED_PTH = os.path.join(cn.TEST_DIR, "classifier")
TEST_DATA_CREATED_PTH = os.path.join(TEST_DATA_CREATED_PTH,
    "test_hypergrid_harness_meta_classifier.csv")


################# HELPERS ##################
def nearAcc(acc1, acc2):
  THR = 0.05
  return np.abs(acc1 - acc2) < THR

def runner(sigma=1.5, num_dim=5, impurity=0.0):
  df = HypergridHarnessMetaClassifier.analyze(
      is_rel=True,  # analyze
      mclf_dct=None,  # HypergridHarnessMetaClassifier
      sigma=sigma, num_repl=3, # MetaClassifier
          # RandomHypergridHarness
      stds=np.repeat(STD, num_dim), impurity=impurity, num_point=NUM_POINT)


################# TESTS ##################
class TestHypergridHarnessMetaClassifier(unittest.TestCase):

  def _cleanUp(self):
    if os.path.isfile(TEST_DATA_PTH):
      os.remove(TEST_DATA_PTH)
  
  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()

  def tearDown(self):
    if os.path.isfile(TEST_DATA_PTH):
      os.remove(TEST_DATA_PTH)

  def _init(self):
    self._cleanUp()
    self.mclf_dct = hypergrid_harness_meta_classifier.MCLF_DCT
    self.harness = HypergridHarnessMetaClassifier(
        mclf_dct=self.mclf_dct,  # HypergridHarnessMetaClassifier
        stds=STDS, impurity=IMPURITY, num_point=NUM_POINT)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self._init()
    if self.harness.df_data is not None:
      self.assertGreater(len(self.harness.trinary.pos_arr), 0)
      self.assertTrue(helpers.isValidDataFrame(self.harness.df_data,
          expected_columns=self.harness.df_data.columns))
    #

  def testEvaluateExperiment(self):
    if IGNORE_TEST:
      return
    self._init()
    score_results = self.harness._evaluateExperiment()
    self.assertEqual(len(score_results), len(self.harness.mclf_dct))
    self.assertTrue(nearAcc(score_results[IDX_PLURALITY].abs,
        SCORE_PLURALITY))
    trues = [score_results[i].abs > score_results[IDX_PLURALITY].abs
        for i in range(len(score_results)) if i != IDX_PLURALITY]
    self.assertTrue(all(trues))

  def testAnalyze(self): 
    if IGNORE_TEST:
      return
    sigma = 2.0
    num_dim = 2
    impurity = -0.6
    df = HypergridHarnessMetaClassifier.analyze(
        # MetaClassifier
        mclf_dct=hypergrid_harness_meta_classifier.MCLF_DCT,
        is_rel=False, iter_count=10,
        sigma=sigma, num_repl=NUM_REPL,
        # RandomHypergridHarness
        stds=np.repeat(STD, num_dim), impurity=impurity,
        num_point=NUM_POINT)
    self.assertTrue(helpers.isValidDataFrame(df,
        expected_columns=["policy", cn.MEAN, cn.STD, cn.COUNT]))

  def testMakeEvaluationData(self):
    if IGNORE_TEST:
      return
    self._cleanUp()
    HypergridHarnessMetaClassifier.makeEvaluationData(
        is_quiet=True,
        is_test=True, out_pth=TEST_DATA_PTH)
    self.assertTrue(os.path.isfile(TEST_DATA_PTH))

  def testPlotMetaClassifiers(self):
    if IGNORE_TEST:
      return
    # Smoke test
    self._init()
    harness = HypergridHarnessMetaClassifier(
        mclf_dct=self.mclf_dct,  # HypergridHarnessMetaClassifier
        data_pth=TEST_DATA_CREATED_PTH,
        stds=STDS, impurity=IMPURITY, num_point=NUM_POINT)
    harness.plotMetaClassifiers(5, 0.2, 0.0, is_plot=IS_PLOT,
        title="This is fine.", xlim=[0, 1])
    harness.plotMetaClassifiers(15, 0.6, -0.6,
        xlim=[0,1],  is_plot=IS_PLOT)
    harness.plotMetaClassifiers(15, 0.6, -0.84,
        clf="logistic",
        xlim=[0,1], title="logistic", is_plot=IS_PLOT)

  def testPlotMultipleMetaClassifiers(self):
    if IGNORE_TEST:
      return
    # Smoke test
    stdw = 0.2
    clf = "SVM"
    self._init()
    harness = HypergridHarnessMetaClassifier(
        mclf_dct=self.mclf_dct,  # HypergridHarnessMetaClassifier
        data_pth=TEST_DATA_CREATED_PTH,
        stds=STDS, impurity=IMPURITY, num_point=NUM_POINT)
    impuritys = list(harness.df_data["impurity"].unique())
    impuritys.sort()
    harness.plotMultipleMetaClassifiers(5, stdw, impuritys,
        figsize=(10, 12), clf=clf,
        is_plot=IS_PLOT)
    harness.plotMultipleMetaClassifiers(15, stdw, impuritys,
        figsize=(10, 12), clf=clf,
        is_plot=IS_PLOT)

  def testAnalyzeLogisticRegressionClassifiers(self):
    if IGNORE_TEST:
      return
    mclf_dct = {
        "plurality": MetaClassifierPlurality(),
        "default":
         MetaClassifierDefault(
         clf=LogisticRegression(random_state=0)),
        "augment":
         MetaClassifierAugment(
         clf=LogisticRegression(random_state=0)),
        "average":
         MetaClassifierAverage(
         clf=LogisticRegression(random_state=0)),
        "ensemble":
         MetaClassifierEnsemble(
         clf=LogisticRegression(random_state=0))
        }
    sigma = 0
    num_dim = 2
    impurity = -0.6
    df = HypergridHarnessMetaClassifier.analyze(
        # MetaClassifier
        mclf_dct=mclf_dct,
        is_rel=False, iter_count=10,
        sigma=sigma, num_repl=NUM_REPL,
        # RandomHypergridHarness
        stds=np.repeat(STD, num_dim), impurity=impurity,
        num_point=NUM_POINT)
    self.assertTrue(helpers.isValidDataFrame(df,
        expected_columns=["policy", cn.MEAN, cn.STD, cn.COUNT]))


if __name__ == '__main__':
  unittest.main()
