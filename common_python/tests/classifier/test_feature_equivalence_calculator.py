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
CLS_FEATURES = ["Rv0158", "Rv1460"]
CLS_FEATURE = CLS_FEATURES[0]
NUM_CMP_FEATURES = 3
CMP_FEATURES = DF_X.columns.tolist()
CMP_FEATURES =  \
    CMP_FEATURES[:NUM_CMP_FEATURES]
FEATURES = list(CMP_FEATURES)
FEATURES.extend(CLS_FEATURES)
IDX = 1
FIT_RESULT = mcfo.FitResult(
    idx=IDX, sels=CLS_FEATURES, sels_score=0.7,
    all_score=1, excludes=[], n_eval=1)
FIT_RESULT2 = mcfo.FitResult(
    idx=IDX+1, sels=CLS_FEATURES, sels_score=0.7,
    all_score=1, excludes=[], n_eval=1)
NUM_CROSS_ITER = 5
NUM_CROSS_ITER_ACCURATE = 50


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
    cmp_features = list(CMP_FEATURES)
    cmp_features.insert(0, CLS_FEATURE)
    calculator =  \
        fec.FeatureEquivalenceCalculator(self.df_X,
        self.ser_y,
        num_cross_iter=NUM_CROSS_ITER_ACCURATE)
    df_ria = calculator._calculateRIA(
        CLS_FEATURES, cmp_features)
    helpers.isValidDataFrame(df_ria, 
        expected_columns=[cn.CLS_FEATURE, cn.CMP_FEATURE,
        cn.SCORE], nan_columns=[cn.SCORE])
    for _, row in df_ria.iterrows():
      if row[cn.CMP_FEATURE] == row[cn.CLS_FEATURE]:
        self.assertEqual(row[cn.SCORE], 1)
      else:
        self.assertFalse(np.isnan(row[cn.SCORE]))

  def testRunOne(self):
    if IGNORE_TEST:
      return
    self._init()
    calculator =  \
        fec.FeatureEquivalenceCalculator(self.df_X,
        self.ser_y,
        num_cross_iter=NUM_CROSS_ITER_ACCURATE)
    calculator.run([FIT_RESULT])
    df = calculator.df_ria
    for feature in CLS_FEATURES:
      self.assertEqual(len(df[df[cn.CLS_FEATURE]
          == feature]), len(FIT_RESULT.sels))
      dff = df[df[cn.CLS_FEATURE] == feature]
      dff = dff[dff[cn.CMP_FEATURE] == feature]
      score = dff[cn.SCORE].values[0]
      self.assertEqual(score, 1)
    #
    columns = [cn.SCORE, cn.CLS_FEATURE, cn.CMP_FEATURE]
    self.assertTrue(helpers.isValidDataFrame(df, columns))

  def testRunMany(self):
    if IGNORE_TEST:
      return
    self._init()
    fit_results = [FIT_RESULT, FIT_RESULT]
    self.calculator.run(fit_results)
    num_sels = len(FIT_RESULT.sels)
    self.assertEqual((2*num_sels)**2,
        len(self.calculator.df_ria))
    

if __name__ == '__main__':
  unittest.main()
