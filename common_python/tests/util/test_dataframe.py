import pandas as pd
import numpy as np
import unittest

import common_python.util.dataframe as dataframe

IGNORE_TEST = False
IS_PLOT = False
COL_A = 'a'
COL_B = 'b'
COL_C = 'c'
DF = pd.DataFrame({COL_A: range(3)})
DF[COL_B] = 10*DF[COL_A]
SIZE = 3
DFS = [DF for _ in range(SIZE)]
DF1 = pd.DataFrame({COL_A: range(SIZE), COL_B: range(SIZE)})
DF2 = pd.DataFrame({COL_A: range(SIZE), COL_C: range(SIZE)})
DF1.index = [10, 20, 30]
DF2.index = [10, 30, 40]


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

  def testIntersection(self):
    if IGNORE_TEST:
      return
    def test(axis, predicate):
      df = DF1.copy()
      if axis == 1:
        items = DF2.columns
      else:
        items = DF2.index
      df = dataframe.subset(df, items, axis=axis)
      self.assertTrue(predicate(df))
    #
    predicate = lambda df: (len(df.columns) == 1) and (len(df) == SIZE)
    test(1, predicate)
    predicate = lambda df: (len(df.columns) == 2) and (len(df) == SIZE - 1)
    test(0, predicate)
      


if __name__ == '__main__':
  unittest.main()
