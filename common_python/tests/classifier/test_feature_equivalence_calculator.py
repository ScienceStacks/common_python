import common_python.constants as cn
from common_python.util.persister import Persister
from common_python.testing import helpers
from common_python.classifier  \
    import feature_equivalence_calculator as fec
from common_python.tests.classifier import helpers as test_helpers
from common_python.testing import helpers

import copy
import os
import pandas as pd
import numpy as np
from sklearn import svm
import unittest

IGNORE_TEST = True
DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PERSISTER_PATH = os.path.join(DIR,
    "test_feature_eqivalence_calculator.pcl")
CLASS = 1
DF_X, SER_Y = test_helpers.getDataLong()
SER_Y = pd.Series([
    cn.PCLASS if v == CLASS else cn.NCLASS
    for v in SER_Y], index=SER_Y.index)


class TestFeatureEquivalenceCalculator(unittest.TestCase):
  
  def _init(self):
    self.df_X = copy.deepcopy(DF_X)
    self.ser_y = copy.deepcopy(SER_Y)
    self.calculator =  \
        fec.FeatureEquivalenceCalculator(DF_X, SER_Y)
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
        fec.NUM_CROSS_ITER)

  def testCalculateRIA(self):
    # TESTING
    self._init()
    SEL_FEATURE = "Rv2034"
    features = [
        SEL_FEATURE,
        "Rv2009", "Rv0022c", "Rv3167c",
        SEL_FEATURE,
        ]
    SIZE_SEL = 2
    selected_features = features[:SIZE_SEL]
    selected_feature = selected_features[0]
    alternative_features = features[SIZE_SEL:]
    scores = self.calculator._calculateRIA(
        selected_feature,
        selected_features, alternative_features)
    idx_sel = alternative_features.index(SEL_FEATURE)
    for idx, score in enumerate(scores):
      if idx == idx_sel:
        self.assertEqual(score, 1)
      else:
        self.assertGreater(np.abs(score - 1), 0.1)


if __name__ == '__main__':
  unittest.main()
