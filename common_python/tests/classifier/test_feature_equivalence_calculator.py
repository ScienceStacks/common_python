import common_python.constants as cn
from common_python.util.persister import Persister
from common_python.testing import helpers
from common_python.classifier  \
    import feature_equivalence_calculator as fec
from common_python.classifier  \
    import multi_classifier_feature_optimizer as mcfo
from common_python.tests.classifier import helpers as test_helpers
from common_python.testing import helpers

import copy
import os
import pandas as pd
import numpy as np
from sklearn import svm
import unittest

IGNORE_TEST = False
DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PERSISTER_PATH = os.path.join(DIR,
    "test_feature_eqivalence_calculator.pcl")
CLASS = 1
DF_X, SER_Y = test_helpers.getDataLong()
SER_Y = pd.Series([
    cn.PCLASS if v == CLASS else cn.NCLASS
    for v in SER_Y], index=SER_Y.index)
SELECTED_FEATURES = ["Rv2034", "Rv2009"]
SELECTED_FEATURE = SELECTED_FEATURES[0]
NUM_ALTERNATIVE_FEATURES = 3
ALTERNATIVE_FEATURES = DF_X.columns.tolist()
ALTERNATIVE_FEATURES =  \
    ALTERNATIVE_FEATURES[:NUM_ALTERNATIVE_FEATURES]
FEATURES = list(ALTERNATIVE_FEATURES)
FEATURES.extend(SELECTED_FEATURES)
IDX = 1
FIT_RESULT = mcfo.FitResult(
    idx=IDX, sels=SELECTED_FEATURES, sels_score=0.7,
    all_score=1, excludes=[], n_eval=1)
FIT_RESULT2 = mcfo.FitResult(
    idx=IDX+1, sels=SELECTED_FEATURES, sels_score=0.7,
    all_score=1, excludes=[], n_eval=1)
NUM_CROSS_ITER = 2


class TestFeatureEquivalenceCalculator(unittest.TestCase):
  
  def _init(self):
    self.df_X = copy.deepcopy(DF_X)
    self.features = list(FEATURES)
    self.df_X = self.df_X[self.features]
    self.ser_y = copy.deepcopy(SER_Y)
    self.calculator =  \
        fec.FeatureEquivalenceCalculator(self.df_X,
        self.ser_y, num_cross_iter=NUM_CROSS_ITER)
    self._remove()

  def _remove(self):
    if os.path.isfile(TEST_PERSISTER_PATH):
      os.remove(TEST_PERSISTER_PATH)

  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()

  def tearDown(self):
    self._remove()

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(len(self.calculator._partitions),
        NUM_CROSS_ITER)

  def testCalculateRIA(self):
    if IGNORE_TEST:
      return
    self._init()
    scores = self.calculator._calculateRIA(
        SELECTED_FEATURE,
        SELECTED_FEATURES, ALTERNATIVE_FEATURES)
    idx_sel = SELECTED_FEATURES.index(SELECTED_FEATURE)
    for idx, score in enumerate(scores):
      if idx == idx_sel:
        self.assertEqual(score, 1)
      else:
        self.assertFalse(np.isnan(score))

  def testRunOne(self):
    if IGNORE_TEST:
      return
    self._init()
    self.calculator.run([FIT_RESULT])
    for feature in SELECTED_FEATURES:
      df = self.calculator.ria_dct[IDX]   
      self.assertEqual(len(df[df[fec.SELECTED_FEATURE]
          == feature]), len(FIT_RESULT.sels))
      dff = df[df[fec.SELECTED_FEATURE] == feature]
      dff = dff[dff[fec.ALTERNATIVE_FEATURE] == feature]
      score = dff[cn.SCORE].values[0]
      self.assertEqual(score, 1)
    #
    columns = [cn.SCORE, fec.SELECTED_FEATURE,
        fec.ALTERNATIVE_FEATURE]
    self.assertTrue(helpers.isValidDataFrame(df, columns))

  def testRunMany(self):
    if IGNORE_TEST:
      return
    self._init()
    fit_results = [FIT_RESULT, FIT_RESULT2]
    self.calculator.run(fit_results)
    self.assertEqual(len(self.calculator.ria_dct),
        len(fit_results))
    
    


if __name__ == '__main__':
  unittest.main()
