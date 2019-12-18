import pandas as pd
import unittest

import common_python.util.dataframe as dataframe

COL_A = 'a'
COL_B = 'b'
DF = pd.DataFrame({COL_A: range(3)})
DF[COL_B] = 10*DF[COL_A]


class TestFunctions(unittest.TestCase):

  def testisLessEqual(self):
    df2 = DF.applymap(lambda v: v - 1)
    self.assertTrue(dataframe.isLessEqual(df2, DF))
    self.assertFalse(dataframe.isLessEqual(DF, df2))
    self.assertTrue(dataframe.isLessEqual(DF, DF))


if __name__ == '__main__':
  unittest.main()
