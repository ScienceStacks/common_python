import common_python.constants as cn
from common_python.testing import helpers
from common_python.tests.classifier import helpers as test_helpers
from common_python.classsifier import multi_classifier
from common_python.testing import helpers

import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = False


class TestFeatureHandler(unittest.TestCase):
  
  def setUp(self):
    self.df_X, self.ser_y = test_helpers.getDataLong()
    self.handler = multi_classifier.FeatureHandler(
        self.df_X, self.ser_y)
    

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertTrue(self.df_X.equals(
        self.handler.df_X))
    import pdb; pdb.set_trace()


class TestMultiClassifier(unittest.TestCase):
  
  def setUp(self):
    pass

  def testConstructor(self):
    if IGNORE_TEST:
      return


if __name__ == '__main__':
  unittest.main()


