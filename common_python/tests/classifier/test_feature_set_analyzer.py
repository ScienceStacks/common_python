
from common_python.tests.classifier import helpers as test_helpers
from common_python.classifier.feature_set_analyzer import FeatureSetAnalyzer

import os
import unittest

IGNORE_TEST = False
IS_PLOT = True
CLASS = 1
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR_PATH = os.path.join(TEST_DIR,
                           "test_feature_analyzer_%d" % CLASS)
ANALYZER = test_helpers.getFeatureAnalyzer()


class TestFeatureSet(unittest.TestCase):

  def setUp(self):
    self.fs_analyzer = FeatureSetAnalyzer(ANALYZER)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertGreater(len(self.fs_analyzer.ser_fset), 0)


if __name__ == '__main__':
  unittest.main()


