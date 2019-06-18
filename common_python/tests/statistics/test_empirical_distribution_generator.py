"""Tests for Statistics Utilities."""

from common_python.statistics import empirical_distribution_generator
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


class TestEmpiricalDistributionGenerator(unittest.TestCase):

  def setUp(self):
    self.cls = empirical_distribution_generator.EmpiricalDistributionGenerator
    self.empirical = self.cls(DF)

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
    df = self.cls.decorrelate(
        df_orig)
    self.assertTrue(helpers.isValidDataFrame(df, df_orig.columns))

  def testSynthesize(self):
    return

if __name__ == '__main__':
  unittest.main()
