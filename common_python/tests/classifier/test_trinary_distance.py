from common import constants as cxn
from common_python.classifier.trinary_distance import TrinaryDistance

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = False
IS_PLOT = False
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
    if IGNORE_TEST:
      return
    self.tdistance.calcDistance()
    for col1, col2 in [ (COL_A, COL_C), (COL_C, COL_B) ]:
        self.assertEqual(self.tdistance.df_distance.loc[col1, col2], 4)
    self.assertEqual(np.shape(self.tdistance.df_distance.values), (3, 3))

  def testCalcDistanceScale(self):
    if IGNORE_TEST:
      return
    column_vecs = []
    size = 1000
    for _ in range(size):
        column_vecs.append(np.random.randint(0, 3, size))
    df_trinary = pd.DataFrame(column_vecs)
    df_trinary = df_trinary.applymap(lambda v: v - 1)
    tdistance = TrinaryDistance(df_trinary)
    tdistance.calcDistance()
    diag = np.diagonal(tdistance.df_distance)
    self.assertTrue(np.isclose(sum(diag*diag), 0))
    # Expect a high density around 36
    _ = plt.hist(tdistance.df_distance.values.flatten())
    if IS_PLOT:
      plt.show()


if __name__ == '__main__':
  unittest.main()
