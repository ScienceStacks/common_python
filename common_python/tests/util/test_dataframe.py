import pandas as pd
import numpy as np
import unittest

import common_python.util.dataframe as dataframe

IGNORE_TEST = False
COL_A = 'a'
COL_B = 'b'
DF = pd.DataFrame({COL_A: range(3)})
DF[COL_B] = 10*DF[COL_A]
SIZE = 3
DFS = [DF for _ in range(SIZE)]


class TestFunctions(unittest.TestCase):

  def testisLessEqual(self):
    if IGNORE_TEST:
      return
    df2 = DF.applymap(lambda v: v - 1)
    self.assertTrue(dataframe.isLessEqual(df2, DF))
    self.assertFalse(dataframe.isLessEqual(DF, df2))
    self.assertTrue(dataframe.isLessEqual(DF, DF))

  def testMean(self):
    if IGNORE_TEST:
      return
    df_mean = dataframe.mean(DFS)
    df_mean = df_mean.applymap(lambda v: int(v))
    self.assertTrue(df_mean.equals(DF))

  def testStd(self):
    if IGNORE_TEST:
      return
    df_std = dataframe.std(DFS)
    df_falses = df_std.applymap(lambda v: not np.isclose(v, 0))
    self.assertEqual(df_falses.sum().sum(), 0)


if __name__ == '__main__':
  unittest.main()
