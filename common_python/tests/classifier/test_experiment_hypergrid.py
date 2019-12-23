from common_python.classifier.experiment_hypergrid  \
    import ExperimentHypergrid, TrinaryClassification
from common_python.testing import helpers
import common_python.constants as cn
from common_python.tests.classifier import helpers as test_helpers

import pandas as pd
import numpy as np
from sklearn import svm
import unittest

IGNORE_TEST = False
IS_PLOT = False
POS_ARRS = np.array([ [1, 1], [1, 0], [0, 1] ])
NEG_ARRS = np.array([ [-1, -1], [-1, 0], [0, -1] ])
OTHER_ARRS = np.array([ [0, 0] ])


class TestTrinaryClassification(unittest.TestCase):

  def setUp(self):
    self.trinary = TrinaryClassification(
        pos_arrs=POS_ARRS,
        neg_arrs=NEG_ARRS,
        other_arrs=OTHER_ARRS)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(self.trinary.dim, 2)

  def testPerturb(self):
    if IGNORE_TEST:
      return
    trinary = self.trinary.perturb(0)
    for arr_type in ["pos", "neg", "other"]:
      arrs1 = eval("self.trinary.%s_arrs" % arr_type)
      arrs2 = eval("trinary.%s_arrs" % arr_type)
      num_trues = sum(sum([a1 == a2 for a1, a2 in zip(arrs1, arrs2)]))
      self.assertEqual(num_trues, 2*len(arrs1))
    #
    unperturb_sum = sum(sum(self.trinary.pos_arrs))
    perturb_sum = 0
    NUM_REPEATS = 30
    SIGMA = 0.1
    for _ in range(NUM_REPEATS):
      trinary = self.trinary.perturb(SIGMA)
      perturb_sum += sum(sum((trinary.pos_arrs)))
    perturb_sum = perturb_sum / NUM_REPEATS
    max_diff = 2*SIGMA/np.sqrt(NUM_REPEATS)
    self.assertLess(np.abs(perturb_sum-unperturb_sum), max_diff)


class TestExperimentHypergrid(unittest.TestCase):

  def setUp(self):
    self.experiment = ExperimentHypergrid()

  def testConstructor(self):
    if IGNORE_TEST:
      return
    tot = sum([len(v) for v in 
        [self.experiment.trinary.neg_arrs,
        self.experiment.trinary.pos_arrs,
        self.experiment.trinary.other_arrs]])  \
        *self.experiment._num_dim
    self.assertEqual(len(self.experiment.grid), 2)
    self.assertEqual(np.size(self.experiment.grid), tot)

  def testPlotGrid(self):
    if IGNORE_TEST:
      return
    # Smoke test
    self.experiment.plotGrid(is_plot=IS_PLOT)
    # With perturbation
    trinary = self.experiment.perturb(0.5)
    self.experiment.plotGrid(trinary=trinary, is_plot=IS_PLOT)

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
