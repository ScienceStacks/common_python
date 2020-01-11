from common_python.classifier.hypergrid_harness  \
    import TrinaryClassification, Plane, Vector
from common_python.classifier import random_hypergrid_harness
from common_python.classifier.random_hypergrid_harness  \
    import RandomHypergridHarness
from common_python.classifier.meta_classifier  \
    import MetaClassifierDefault, MetaClassifierPlurality
from common_python.testing import helpers
import common_python.constants as cn
from common_python.tests.classifier import helpers as test_helpers

import pandas as pd
import numpy as np
from sklearn import svm
import unittest

NUM_DIM = 2
IGNORE_TEST = False
STDS = [0.1, 0.1]
IMPURITY = 0
NUM_POINT = 25


class TestRandomHypergridHarness(unittest.TestCase):
  
  def setUp(self):
    self.harness = RandomHypergridHarness(NUM_POINT,
        STDS, IMPURITY)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    def test(impurity):
      harness = RandomHypergridHarness(NUM_POINT, STDS, impurity)
      self.assertLess(np.abs(impurity - harness.trinary.impurity),
          random_hypergrid_harness.THR_IMPURITY)
    #
    for impurity in [0, -0.6, -0.8, 0.1]:
      test(impurity)


if __name__ == '__main__':
  unittest.main()
