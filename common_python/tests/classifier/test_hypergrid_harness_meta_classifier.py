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

import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = False
IS_PLOT = False
NUM_DIM = 2
DENSITY = 10
OFFSET = 0
IDX_PLURALITY = 0
IDX_DEFAULT = 1
IDX_AUGMENT = 2
IDX_AVERAGE = 3
IDX_ENSEMBLE = 4
SCORE_PLURALITY = 0.5
SCORE_IDEAL = 1.0


################# HELPERS ##################
def runner(sigma=1.5, num_dim=5, impurity=0.0):
  df = HypergridHarnessMetaClassifier.analyze(mclf_dct=None,
      sigmas=sigma, num_dim=num_dim, impurity=impurity,
      num_repl=3,  num_point=25, density=10, is_rel=True)

def testAnalyze():
  sigma = 2.0
  num_dim = 8
  impurity = 0
  df = HypergridHarnessMetaClassifier.analyze(
      mclf_dct=hypergrid_harness_meta_classifier.MCLF_DCT,
      sigma=sigma, num_dim=num_dim, impurity=impurity,
      iter_count=1,
      num_repl=3,  num_point=25, density=10, is_rel=False)


################# TESTS ##################
class TestHypergridHarnessMetaClassifier(unittest.TestCase):
  
  def setUp(self):
    self.mclf_dct = mclf_dct=hypergrid_harness_meta_classifier.MCLF_DCT
    self._init()

  def _init(self):
    self.harness = HypergridHarnessMetaClassifier(
        self.mclf_dct, density=DENSITY)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(self.harness._density, DENSITY)
    self.assertGreater(len(self.harness.trinary.pos_arr), 0)
    self.assertTrue(helpers.isValidDataFrame(self.harness.df_data,
        expected_columns=self.harness.df_data.columns))
    #

  def testEvaluateExperimentSingle(self):
    if IGNORE_TEST:
      return
    score_results = self.harness._evaluateExperiment()
    self.assertEqual(len(score_results), len(self.harness.mclf_dct))
    self.assertEqual(score_results[IDX_PLURALITY].abs,
        SCORE_PLURALITY)
    trues = [score_results[i].abs == SCORE_IDEAL
        for i in range(len(score_results)) if i != IDX_PLURALITY]
    self.assertTrue(all(trues))

  def testEvaluateExperimentMultiple(self):
    return
    if IGNORE_TEST:
      return
    NUM_REPL = 3
    SIGMA = 1.5
    NUM_DIM = 15
    num_point = 25
    vector = Vector(np.repeat(1, NUM_DIM))
    plane = Plane(vector)
    harness = HypergridHarnessMetaClassifier(
        self.mclf_dct, density=1.5, plane=plane,
        num_point=num_point, impurity=0)
    rel_scoress = []
    for _ in range(1000):
      try:
        score_results = harness._evaluateExperiment(
            sigma=SIGMA, num_repl=NUM_REPL)
        rel_scores = [score_results[i].rel
            for i in range(len(score_results)) if i != IDX_PLURALITY]
        rel_scoress.append(rel_scores)
      except:
        pass
    arr = np.array(rel_scoress)
    df = pd.DataFrame(arr)
    ser_mean = df.mean()
    ser_std = df.std() / np.sqrt(len(rel_scoress))
    harness.plotGrid(
        trinary=harness.trinary.perturb(sigma=SIGMA)[0],
        xlim=[-5,5], ylim=[-5,5])
    self.assertLess(np.std(rel_scores), 0.01)

  def testAnalyze(self): 
    if IGNORE_TEST:
      return
    sigma = 2.0
    num_dim = 2
    impurity = 0
    df = HypergridHarnessMetaClassifier.analyze(
        mclf_dct=hypergrid_harness_meta_classifier.MCLF_DCT,
        sigma=sigma, num_dim=num_dim, impurity=impurity,
        iter_count=2, is_iter_report=False,
        num_repl=3,  num_point=25, density=10, is_rel=False)
    self.assertTrue(helpers.isValidDataFrame(df,
        expected_columns=["policy", cn.MEAN, cn.STD, cn.COUNT]))


if __name__ == '__main__':
  unittest.main()

