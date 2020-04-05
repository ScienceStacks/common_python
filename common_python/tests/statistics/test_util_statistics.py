"""Tests for Statistics Utilities."""

from common_python.statistics import util_statistics

import numpy as np
import pandas as pd
import scipy.stats
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
NZ_INDICES = [i for i in DF.index if i[0:2] == 'nz']
Z_INDICES = [i for i in DF.index if i[0] == 'z']


class TestFunctions(unittest.TestCase):

  def testFilterZeroVarianceRows(self):
    if IGNORE_TEST:
      return
    df = util_statistics.filterZeroVarianceRows(DF)
    difference = set(NZ_INDICES).symmetric_difference(df.index)
    self.assertEqual(len(difference), 0)

  def testCalcLogSL(self):
    if IGNORE_TEST:
      return
    df = util_statistics.calcLogSL(DF)
    for index in Z_INDICES:
      trues = [np.isnan(v) for v in df.loc[index, :]]
    self.assertTrue(all(trues))
    #
    columns = df.columns
    for index in NZ_INDICES:
      for nn in range(2, len(df.loc[index, :])):
        self.assertLess(df.loc[index, columns[nn-1]],
            df.loc[index, columns[nn]])

    def testDecorelate(self):
      if IGNORE_TEST:
        return
      df_orig = pd.concat([DF for _ in range(500)], axis=1)
      df_orig.columns = [
          "%d" % d for d in range(len(df_orig.columns))]
      df = util_statistics.decorrelate(df_orig)
      self.assertTrue(helpers.isValidDataFrame(df, df_orig.columns))

  def testGeneralizedBinomialDensity(self):
    if IGNORE_TEST:
      return
    def test(size, num_choose, prob):
      probs = np.repeat(prob, size)
      result = util_statistics.generalizedBinomialDensity(
          probs, num_choose)
      expected = scipy.stats.binom.pmf(num_choose,
          size, prob)
      self.assertTrue(np.isclose(result, expected))
    #
    test(4, 0, 0.25)
    test(4, 4, 0.25)
    test(25, 5, 0.02)
    test(4, 2, 0.25)
    test(8, 2, 0.2)
    test(10, 2, 0.02)

  def testGeneralizedBinomialDensity2(self):
    if IGNORE_TEST:
      return
    PROBS = [0.25, 0.2, 0.2, 0.2]
    prob = 0.25
    def test(num_choose):
      probs = [0.25, 0.2, 0.2, 0.2]
      result = util_statistics.generalizedBinomialDensity(
          PROBS, num_choose)
      expected = scipy.stats.binom.pmf(num_choose,
          len(PROBS), prob)
      import pdb; pdb.set_trace()
      self.assertTrue(np.isclose(result, expected))
    #
    test(len(PROBS))

  def testGeneralizedBinomialDensity2(self):
    if IGNORE_TEST:
      return
    LOW_PROB = 0.2
    HIGH_PROB = 0.25
    SIZE = 4
    PROBS = np.repeat(LOW_PROB, SIZE-1)
    PROBS = PROBS.tolist()
    PROBS.insert(0, HIGH_PROB)
    #
    total = 0
    for n in range(SIZE + 1):
      prob = util_statistics.generalizedBinomialDensity(
          PROBS, n)
      total += prob
    self.assertTrue(np.isclose(total, 1.0))
    low_prob = util_statistics.generalizedBinomialDensity(
        PROBS, SIZE)
    high_prob = scipy.stats.binom.pmf(SIZE,
        SIZE, HIGH_PROB)
    self.assertLess(low_prob, high_prob)

  def testGeneralizedBinomialTail(self):
    if IGNORE_TEST:
      return
    def test(size, num_choose, prob):
      probs = np.repeat(prob, size)
      result = util_statistics.generalizedBinomialTail(
          probs, num_choose)
      expected = scipy.stats.binom.cdf(num_choose,
          size, prob)
      expected = 1 - expected + scipy.stats.binom.pmf(
          num_choose, size, prob)
      self.assertTrue(np.isclose(result, expected))
    #
    test(4, 2, 0.25)
    test(4, 0, 0.25)
    test(4, 4, 0.25)
    test(8, 6, 0.2)
    test(10, 5, 0.02)
      

if __name__ == '__main__':
  unittest.main()
