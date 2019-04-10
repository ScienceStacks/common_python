"""Tests for util_plots."""

from common_python.plots import util_plots

import pandas as pd
import unittest

IGNORE_TEST = False
IS_PLOT = False

DF = pd.DataFrame({
  'a': [-1, 0, 1],
  'b': [0, 1, -1],
  'c': [1, -1, 0],
  })


class TestFunction(unittest.TestCase):

  def testPlotTrinaryHeatmap(self):
    # Smoke tests
    if IGNORE_TEST:
      return
    util_plots.plotTrinaryHeatmap(DF, IS_PLOT)


if __name__ == '__main__':
  unittest.main()
