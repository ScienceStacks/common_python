from common import constants as cxn
from common_python.classifier.trinary_distance import TrinaryDistance

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = True
IS_PLOT = True
COL_A = "col_a"
COL_B = "col_b"
COL_C = "col_c"
a_vals = [-1, -1, 0, 0, 1, 1]
b_vals = [-1, -1, 0, 0, 1, 1]
c_vals = [1, 1, 0, 0, -1, -1]
DF = pd.DataFrame({
    COL_A: a_vals,
    COL_B: b_vals,
    COL_C: c_vals,
    })


class TestTrinaryDistance(unittest.TestCase):

  def setUp(self):
    self.tdistance = TrinaryDistance(DF)
  
  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertTrue(self.tdistance.df_trinary.equals(DF))
  
  def testCalcDistance(self):
    # TESTING
    self.tdistance.calcDistance()
    import pdb; pdb.set_trace()


if __name__ == '__main__':
  unittest.main()
