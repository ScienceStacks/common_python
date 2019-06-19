"""Tests for Statistics Utilities."""

from common_python.statistics import density
import common_python.constants as cn
from common_python.testing import helpers

import numpy as np
import pandas as pd
import unittest


IGNORE_TEST = True
IS_PLOT = False
SIZE = 10
MAX_VALUE = 5
COLA = "colA"
values = np.random.randint(1, MAX_VALUE, size=20)
SER = pd.Series(values)
TOLERANCE = 0.001


class TestDensity(unittest.TestCase):

  def setUp(self):
    self.cls = density.Density
    self.density = self.cls(SER)
    
  def testConstructor(self):
    expected = range(1, MAX_VALUE+1)
    self.assertTrue(set(self.density.variates).issubset(expected))

  def testMakeDensity(self):
    return
    

if __name__ == '__main__':
  unittest.main()
