"""Tests for Statistics Utilities."""

from common_python.statistics import util_statistics

import pandas as pd
import unittest

IGNORE_TEST = False
SIZE = 10
DF = pd.DataFrame({
    'nz-1': range(SIZE),
    'z-1': [1 for _ in range(SIZE)],
    'z-2': [2 for _ in range(SIZE)],
    'nz-2': range(SIZE),
})
DF = DF.T


class TestFunctions(unittest.TestCase):

  def testFilterZeroVarianceRows(self):
    df = util_statistics.filterZeroVarianceRows(DF)
    indices = [i for i in DF.index if i[0:2] == 'nz']
    difference = set(indices).symmetric_difference(df.index)
    self.assertEqual(len(difference), 0)


if __name__ == '__main__':
  unittest.main()
