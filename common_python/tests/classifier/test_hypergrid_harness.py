from common_python.classifier.hypergrid_harness  \
    import HypergridHarness, TrinaryClassification, Plane, Vector
from common_python.classifier.meta_classifier  \
    import MetaClassifierDefault, MetaClassifierPlurality
from common_python.testing import helpers
import common_python.constants as cn
from common_python.tests.classifier import helpers as test_helpers

import pandas as pd
import numpy as np
from sklearn import svm
import unittest

NUM_DIM = 2
IGNORE_TEST = False
IS_PLOT = False
NEG_ARRS = np.array([ [-1, -1], [-1, 0], [0, -1] ])
NUM1 = 1.1
NUM2 = 1.2
OTHER_ARRS = np.array([ [0, 0] ])
POS_ARRS = np.array([ [1, 1], [1, 0], [0, 1] ])


class TestVector(unittest.TestCase):

  def testStr(self):
    if IGNORE_TEST:
      return
    vector = Vector(np.array([NUM1, NUM2]))
    stg = str(vector)
    self.assertTrue(str(NUM1) in stg)
    self.assertTrue(str(NUM2) in stg)


class TestPlane(unittest.TestCase):

  def setUp(self):
    self.coef_arr = np.array([1, 1])
    self.offset = 1
    self.plane = Plane(Vector(self.coef_arr), offset=self.offset)

  def testStr(self):
    if IGNORE_TEST:
      return
    vector = Vector(np.array([NUM1, NUM2]))
    plane = Plane(vector, offset=self.offset)
    stg = str(plane)
    self.assertTrue(str(self.offset) in stg)

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
    self.assertEqual(self.trinary.num_dim, 2)

  def testPerturb(self):
    if IGNORE_TEST:
      return
    trinarys = self.trinary.perturb(sigma=0)
    self.assertEqual(len(trinarys), 1)
    trinary = trinarys[0]
    for arr_type in ["pos", "neg", "other"]:
      arrs1 = eval("self.trinary.%s_arr" % arr_type)
      arrs2 = eval("trinary.%s_arr" % arr_type)
      num_trues = sum(sum([a1 == a2 for a1, a2 in zip(arrs1, arrs2)]))
      self.assertEqual(num_trues, 2*len(arrs1))
    #
    SIZE = 3
    trinarys = self.trinary.perturb(sigma=0, num_repl=SIZE)
    self.assertEqual(len(trinarys), SIZE)
    #
    unperturb_sum = sum(sum(self.trinary.pos_arr))
    perturb_sum = 0
    NUM_REPEATS = 300
    SIGMA = 0.1
    for _ in range(NUM_REPEATS):
      trinary = self.trinary.perturb(sigma=SIGMA)[0]
      perturb_sum += sum(sum((trinary.pos_arr)))
    perturb_sum = perturb_sum / NUM_REPEATS
    max_diff = 10*SIGMA/np.sqrt(NUM_REPEATS)
    self.assertLess(np.abs(perturb_sum-unperturb_sum), max_diff)

  def testMakeMatrices(self):
    if IGNORE_TEST:
      return
    df, ser = self.trinary.makeMatrices()
    ser_test = df.sum(axis=1) * ser
    ser_test = ser_test.map(lambda v: v > 0)
    self.assertEqual(len(ser_test), ser_test.sum())

  def testConcat(self):
    if IGNORE_TEST:
      return
    def test(arr1, arr2, num_repl):
      # Count occurrence of members of arr1 in arr2
      for arr in arr1:
        count = len([v for v in arr2 if all(v == arr)])
        self.assertEqual(count, num_repl)
    #
    SIZE =3
    trinary = TrinaryClassification.concat(
        np.repeat(self.trinary, SIZE))
    test(self.trinary.pos_arr, trinary.pos_arr, SIZE)
    test(self.trinary.neg_arr, trinary.neg_arr, SIZE)
    test(self.trinary.other_arr, trinary.other_arr, SIZE)


class TestHypergridHarness(unittest.TestCase):

  def setUp(self):
    self.harness = HypergridHarness(density= 4)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    tot = sum([len(v) for v in 
        [self.harness.trinary.neg_arr,
        self.harness.trinary.pos_arr,
        self.harness.trinary.other_arr]])  \
        *self.harness.num_dim
    self.assertEqual(len(self.harness.grid), 2)
    self.assertEqual(np.size(self.harness.grid), tot)
    #
    num_dim = 4
    plane = Plane(Vector(np.repeat(1, num_dim)))
    harness = HypergridHarness(
        density= 2, plane=plane)
    arr_lst = []
    arr_lst.extend(harness.trinary.pos_arr)
    arr_lst.extend(harness.trinary.neg_arr)
    arr_lst.extend(harness.trinary.other_arr)
    arrs = [tuple(x) for x in arr_lst]
    ser = pd.Series(arrs)
    ser = ser.unique()
    self.assertEqual(len(ser), len(arr_lst))

  def testPlotGrid(self):
    if IGNORE_TEST:
      return
    # Smoke test
    OFFSET = 0.2
    plane = Plane(Vector(np.repeat(1, NUM_DIM)), offset=OFFSET)
    harness = HypergridHarness(density=10, plane=plane)
    # With perturbation
    trinary = harness.perturb(sigma=0.5)[0]
    harness.plotGrid(trinary=trinary, is_plot=IS_PLOT)

  def testEvaluateSVM(self):
    if IGNORE_TEST:
      return
    OFFSET = 1
    plane = Plane(Vector(np.repeat(1, NUM_DIM)), offset=OFFSET)
    harness = HypergridHarness(density=20, plane=plane)
    accuracy, plane = harness.evaluateSVM(sigma=0.0)
    self.assertGreater(accuracy, 0.8)
    self.assertLess(np.abs(plane.offset - OFFSET), 0.1)


if __name__ == '__main__':
  unittest.main()
