"""Tests for Statistics Utilities."""

from common_python.statistics import multiple_density
import common_python.constants as cn
from common_python.testing import helpers

import numpy as np
import pandas as pd
import unittest


def makeData(maxint, size):
  return pd.DataFrame({
      COLA: np.random.randint(1, maxint+1, size),
      COLB: np.random.randint(1, maxint+1, size),
      })

IGNORE_TEST = False
IS_PLOT = False
SIZE = 10
MAX = 5
COLA = "colA"
COLB = "colB"
DF = makeData(MAX, SIZE)


class TestMultipleDensity(unittest.TestCase):

  def setUp(self):
    self.cls = multiple_density.MultipleDensity
    self.multiple= self.cls(DF, range(1, MAX+1))
    import pdb; pdb.set_trace()
    
  def testConstructor(self):
    if IGNORE_TEST:
      return
    

if __name__ == '__main__':
  unittest.main()
