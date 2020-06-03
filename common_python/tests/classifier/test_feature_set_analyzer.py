from common_python.classifier.feature_set_analyzer import FeatureSetAnalyzer
from common_python.classifier import feature_analyzer
from common_python.tests.classifier import helpers as test_helpers

import os
import unittest

IGNORE_TEST = True
IS_PLOT = True
CLASS = 1
DF_X, SER_Y_ALL = test_helpers.getDataLong()
STATES = list(SER_Y_ALL.unique())
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
# Pattern for the serialization directories
SERIAL_PAT = os.path.join(TEST_DIR, "test_feature_analyzer_%d")


class TestFeatureSet(unittest.TestCase):

  def _init(self):
    self.analyzer = feature_analyzer.FeatureAnalyzer.deserialize(
      SERIAL_PAT % CLASS)
    self.s_analyzer = FeatureSetAnalyzer(self.analyzer.ser_fset)

  def setUp(self):
    self._init()

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertGreater(len(self.s_analyzer._ser_fset), 0)

  def testMakeNonIntersectingSets(self):
    if IGNORE_TEST:
      return

if __name__ == '__main__':
  unittest.main()


