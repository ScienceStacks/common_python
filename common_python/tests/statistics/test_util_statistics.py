"""Tests for Statistics Utilities."""

from common_python.statistics import util_statistics

import numpy as np
import pandas as pd
import scipy.stats
import unittest

IGNORE_TEST = True
IS_PLOT = True
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
      r1 = util_statistics.generalizedBinomialDensity(
          probs, num_choose)
      expected = scipy.stats.binom.pmf(num_choose,
          size, prob)
      self.assertTrue(np.isclose(r1, expected))
      #
      r2 = util_statistics.generalizedBinomialDensity(
          probs, num_choose, is_sampled=True)
      diff = np.abs(r1 - r2)
      self.assertTrue(np.isclose(r1, r2))
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

  def testGeneralizedBinomialDensity3(self):
    if IGNORE_TEST:
      return
    def test(size, num_choose, prob):
      probs = np.repeat(prob, size)
      expected = scipy.stats.binom.pmf(num_choose,
          size, prob)
      result = util_statistics.generalizedBinomialDensity(
          probs, num_choose, is_sampled=True)
      diff = np.abs(expected - result)
      self.assertTrue(np.isclose(diff, 0))
    #
    test(20, 4, 0.1)
    test(500, 20, 0.01)

  def testGeneralizedBinomialDensity4(self):
    if IGNORE_TEST:
      return
    SIZE = 10
    PROB_LOW = 0.3
    PROB_HIGH = 0.4
    PROBS = np.repeat(PROB_LOW, SIZE)
    PROBS = np.concatenate([PROBS,
        np.repeat(PROB_HIGH, SIZE)])
    r_exact = util_statistics.generalizedBinomialDensity(
        PROBS, SIZE, is_sampled=False)
    r_smpl = util_statistics.generalizedBinomialDensity(
        PROBS, SIZE, is_sampled=True)
    self.assertLess(np.abs(r_exact - r_smpl), 0.001)

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

  def testGeneralizedBinomialTail2(self):
    # TESTING
    SIZE = 5
    PROB_LOW = 0.3
    PROB_HIGH = 0.4
    PROBS = np.repeat(PROB_LOW, SIZE)
    PROBS = np.concatenate([PROBS,
        np.repeat(PROB_HIGH, SIZE)])
    r_exact = util_statistics.generalizedBinomialTail(
        PROBS, 0, is_sampled=False)
    r_smpl = util_statistics.generalizedBinomialTail(
        PROBS, 0, is_sampled=True)
    self.assertTrue(np.isclose(r_exact, 1))
    self.assertTrue(np.isclose(r_smpl, 1, atol=0.001))

  def testChoose(self):
    if IGNORE_TEST:
      return
    self.assertEqual(util_statistics.choose(10, 3), 120)
    for num_total in range(10, 20):
      for num_choose in range(1, 8):
        r1 = util_statistics.choose(num_total, num_choose)
        r2 = util_statistics.choose(num_total, 
            num_total - num_choose)
        self.assertEqual(r1, r2)
      

if __name__ == '__main__':
  unittest.main()
