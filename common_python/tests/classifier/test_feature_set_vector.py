
from common_python import constants as cn
from common_python.tests.classifier import helpers as test_helpers
from common_python.classifier.feature_set_vector  \
    import FeatureSetVector
from common_python.classifier  \
    import feature_set_vector
from common_python.testing import helpers

import copy
import numpy as np
import os
import shutil
import unittest

IGNORE_TEST = True
IS_PLOT = True
CLASS = 1
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
DF_X, _ = test_helpers.getDataLong()
ANALYZER = test_helpers.getFeatureAnalyzer()
INDEX = "T4.1"
SER_COMB = test_helpers.getCombSer()
MIN_SCORE = 0.7
SER_COMB = SER_COMB[SER_COMB > MIN_SCORE]


class TestFeatureSet(unittest.TestCase):

  def _remove(self):
    pass

  def setUp(self):
    self._remove()
    self.analyzer = copy.deepcopy(ANALYZER)
    self.vector = FeatureSetVector(self.analyzer,
        SER_COMB.index, DF_X.loc[INDEX, :])

  def tearDown(self):
    self._remove()

  def testMake_df_value(self):
    if IGNORE_TEST:
      return
    df_value = self.vector._make_df_value()
    columns=[cn.FEATURE_SET, cn.FEATURE, cn.VALUE]
    self.assertTrue(helpers.isValidDataFrame(
        df_value, expected_columns=columns))

  def test_features(self):
    if IGNORE_TEST:
      return
    true = set(self.vector.features).issubset(
        DF_X.columns)
    self.assertTrue(true)

  def testXor(self):
    # TESTING
    other_vector = copy.deepcopy(self.vector)
    new_vector = self.vector.xor(other_vector)
    ser = new_vector.df_value[cn.VALUE].apply(
        lambda v: np.abs(v))
    self.assertEqual(ser.sum().value, 0)
    # XOR with a 0 vector should yield the original
    ser_X = DF_X.
    ser_X = ser_X.apply(lambda v: 0)
    new_vector = FeatureSetVector(self.analyzer,
        SER_COMB.index, ser_X)
    new_vector = self.vector.xor(other_vector)
    ser = new_vector.df_value[cn.VALUE].apply(
        lambda v: np.abs(v))
    import pdb; pdb.set_trace()
    
    
 



if __name__ == '__main__':
  unittest.main()


