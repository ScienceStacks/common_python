"""Tests for Statistics Utilities."""

from common_python.statistics import empirical_distribution
import common_python.constants as cn
from common_python.testing import helpers

import numpy as np
import pandas as pd
import unittest

IGNORE_TEST = False
IS_PLOT = False
SIZE = 10
COLA = "colA"
COLB = "colB"
DF = pd.DataFrame({
    COLA: range(SIZE),
    COLB: range(SIZE),
    })
MAX_FRAC_DIFF = 0.05
NUM_SAMPLES = 5000
TOLERANCE = 0.001


class TestEmpiricalDistribution(unittest.TestCase):

  def setUp(self):
    self.empirical = empirical_distribution.EmpiricalDistribution(DF)

  def testSample(self):
    if IGNORE_TEST:
      return
    df = self.empirical.sample(NUM_SAMPLES, is_decorrelate=False)
    frac = 1.0/SIZE
    frac_0 = (1.0*len(df[df[COLA] == 0])) / len(df)
    self.assertLess(np.abs(frac - frac_0), MAX_FRAC_DIFF)
    self.assertTrue(helpers.isValidDataFrame(df, DF.columns))
    df = self.empirical.sample(NUM_SAMPLES, is_decorrelate=True)
    self.assertTrue(helpers.isValidDataFrame(df, DF.columns))

  def testDecorelate(self):
    if IGNORE_TEST:
      return
    df_orig = pd.concat([DF for _ in range(500)], axis=1)
    df_orig.columns = ["%d" % d for d in range(len(df_orig.columns))]
    df = empirical_distribution.EmpiricalDistribution.decorrelate(
        df_orig)
    self.assertTrue(helpers.isValidDataFrame(df, df_orig.columns))

  def testPlot(self):
    # Smoke test only
    plot_opts = {cn.PLT_IS_PLOT: IS_PLOT}
    self.empirical._df = self.empirical.__class__.decorrelate(
        self.empirical._df)
    self.empirical.plot(plot_opts=plot_opts)

  def testgetMarginals(self):
    df = empirical_distribution.EmpiricalDistribution.decorrelate(DF)
    empirical = empirical_distribution.EmpiricalDistribution(df)
    df_marginals = empirical.getMarginals()
    df_marginals.index = DF.index
    self.assertTrue(df_marginals.equals(DF))

  def testGetProb(self):
    def test(value):
      prob = self.empirical.getProb(COLA, value)
      expected = (1.0*(value+1)/SIZE)
      expected = max(0, expected)
      expected = min(1, expected)
      self.assertLess(np.abs(prob - prob), TOLERANCE)
    #
    test(6)
    test(9)
    test(0)
    test(20)

  def testSynthesize(self):
    return

if __name__ == '__main__':
  unittest.main()
