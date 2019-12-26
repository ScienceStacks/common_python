from common_python.classifier.experiment_hypergrid  \
    import ExperimentHypergrid, TrinaryClassification, Plane, Vector
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


class TestPlane(unittest.TestCase):

  def setUp(self):
    self.coef_arr = np.array([1, 1])
    self.offset = 1
    self.plane = Plane(Vector(self.coef_arr), offset=self.offset)

  def testComparisons(self):
    if IGNORE_TEST:
      return
    self.assertTrue(self.plane.isLess(np.array([-1, -1])))
    self.assertTrue(self.plane.isGreater(np.array([2, 2])))
    self.assertTrue(self.plane.isNear(np.array([2, -1])))

  def testMakeCoordinates(self):
    if IGNORE_TEST:
      return
    xlim = [-1, 1]
    ylim = xlim
    xv, yv = self.plane.makeCoordinates(xlim, ylim)
    for nn in range(len(self.coef_arr)):
      vec = np.array([xv[nn], yv[nn]])
      self.assertEqual(self.coef_arr.dot(vec), self.offset)


class TestTrinaryClassification(unittest.TestCase):

  def setUp(self):
    self.trinary = TrinaryClassification(
        pos_arr=POS_ARRS,
        neg_arr=NEG_ARRS,
        other_arr=OTHER_ARRS)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(self.trinary.dim, 2)

  def testPerturb(self):
    if IGNORE_TEST:
      return
    trinary = self.trinary.perturb(0)
    for arr_type in ["pos", "neg", "other"]:
      arrs1 = eval("self.trinary.%s_arr" % arr_type)
      arrs2 = eval("trinary.%s_arr" % arr_type)
      num_trues = sum(sum([a1 == a2 for a1, a2 in zip(arrs1, arrs2)]))
      self.assertEqual(num_trues, 2*len(arrs1))
    #
    unperturb_sum = sum(sum(self.trinary.pos_arr))
    perturb_sum = 0
    NUM_REPEATS = 30
    SIGMA = 0.1
    for _ in range(NUM_REPEATS):
      trinary = self.trinary.perturb(SIGMA)
      perturb_sum += sum(sum((trinary.pos_arr)))
    perturb_sum = perturb_sum / NUM_REPEATS
    max_diff = 3*SIGMA/np.sqrt(NUM_REPEATS)
    self.assertLess(np.abs(perturb_sum-unperturb_sum), max_diff)

  def testMakeMatrices(self):
    if IGNORE_TEST:
      return
    df, ser = self.trinary.makeMatrices()
    ser_test = df.sum(axis=1) * ser
    ser_test = ser_test.map(lambda v: v > 0)
    self.assertEqual(len(ser_test), ser_test.sum())


class TestExperimentHypergrid(unittest.TestCase):

  def setUp(self):
    self.experiment = ExperimentHypergrid(num_dim=2,
        density= 4)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    tot = sum([len(v) for v in 
        [self.experiment.trinary.neg_arr,
        self.experiment.trinary.pos_arr,
        self.experiment.trinary.other_arr]])  \
        *self.experiment._num_dim
    self.assertEqual(len(self.experiment.grid), 2)
    self.assertEqual(np.size(self.experiment.grid), tot)
    #
    num_dim = 4
    plane = Plane(Vector(np.repeat(1, num_dim)))
    experiment = ExperimentHypergrid(num_dim=num_dim,
        density= 2, plane=plane)
    arr_lst = []
    arr_lst.extend(experiment.trinary.pos_arr)
    arr_lst.extend(experiment.trinary.neg_arr)
    arr_lst.extend(experiment.trinary.other_arr)
    arrs = [tuple(x) for x in arr_lst]
    ser = pd.Series(arrs)
    ser = ser.unique()
    self.assertEqual(len(ser), len(arr_lst))

  def testPlotGrid(self):
    if IGNORE_TEST:
      return
    # Smoke test
    self.experiment.plotGrid(is_plot=IS_PLOT)
    # With perturbation
    trinary = self.experiment.perturb(0.5)
    self.experiment.plotGrid(trinary=trinary, is_plot=IS_PLOT)

  def testPlotSVM(self):
    linear_svm = svm.LinearSVC()
    #linear_svm.fit(


if __name__ == '__main__':
  unittest.main()
