from common_python.classifier.experiment_hypergrid  \
    import ExperimentHypergrid
from common_python.testing import helpers
import common_python.constants as cn
from common_python.tests.classifier import helpers as test_helpers

import pandas as pd
import numpy as np
from sklearn import svm
import unittest

IGNORE_TEST = False
IS_PLOT = False


class TestExperimentHypergrid(unittest.TestCase):

  def setUp(self):
    self.experiment = ExperimentHypergrid()

  def testConstructor(self):
    if IGNORE_TEST:
      return
    tot = sum([len(v) for v in 
        [self.experiment.neg_arrs, self.experiment.pos_arrs,
        self.experiment.other_arrs]])*self.experiment._num_dim
    self.assertEqual(len(self.experiment.grid), 2)
    self.assertEqual(np.size(self.experiment.grid), tot)

  def testPlotGrid(self):
    if IGNORE_TEST:
      return
    # Smoke test
    self.experiment.plotGrid(is_plot=IS_PLOT)

  def testMakePlotValues(self):
    if IGNORE_TEST:
      return
    xlim = [-1, 1]
    ylim = xlim
    vector = np.array([2, 1])
    xv, yv = self.experiment._makePlotValues(vector, xlim, ylim)
    for nn in range(len(vector)):
      vec = np.array([xv[nn], yv[nn]])
      self.assertEqual(vector.dot(vec), 0)


if __name__ == '__main__':
  unittest.main()
