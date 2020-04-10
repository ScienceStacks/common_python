from common_python.classifier import util_classifier
from common_python.tests.classifier  \
    import helpers as test_helpers

import pandas as pd
import numpy as np
import unittest


IGNORE_TEST = True
SER_Y = pd.Series({
    "0-1": 0,
    "0-2": 0,
    "1-1": 1,
    "1-2": 1,
    "2-1": 2,
    "2-2": 2,
    "2-3": 2,
    })
STATE_PROBS = pd.Series([
    2/len(SER_Y),
    2/len(SER_Y),
    3/len(SER_Y),
    ])

DF_PRED = pd.DataFrame({
  0: [0.8, 0, 0.2],
  1: [0, 0.81, 0.19],
  2: [0, 0, 1],
  })
DF_PRED = DF_PRED.T


class TestFunctions(unittest.TestCase):

  def setUp(self):
    self.df_X_short, self.ser_y_short =  \
        test_helpers.getData()
    self.df_X_long, self.ser_y_long =  \
        test_helpers.getDataLong()
 
  def testFindAdjacentStates(self):
    if IGNORE_TEST:
      return
    adjacents = util_classifier.findAdjacentStates(
        SER_Y, "0-1")
    self.assertTrue(np.isnan(adjacents.prv))
    self.assertTrue(np.isnan(adjacents.p_dist))
    self.assertEqual(adjacents.nxt, 1)
    self.assertEqual(adjacents.n_dist, 2)
    #
    adjacents = util_classifier.findAdjacentStates(
        SER_Y, "2-3")
    self.assertTrue(np.isnan(adjacents.nxt))
    self.assertTrue(np.isnan(adjacents.n_dist))
    self.assertEqual(adjacents.prv, 1)
    self.assertEqual(adjacents.p_dist, 3)
    #
    adjacents = util_classifier.findAdjacentStates(
        SER_Y, "1-2")
    self.assertEqual(adjacents.nxt, 2)
    self.assertEqual(adjacents.n_dist, 1)
    self.assertEqual(adjacents.prv, 0)
    self.assertEqual(adjacents.p_dist, 2)

  def testCalcStateProbs(self):
    if IGNORE_TEST:
      return
    ser = util_classifier.calcStateProbs(SER_Y)
    self.assertTrue(ser.equals(STATE_PROBS))

  def testAggregatePredications(self):
    if IGNORE_TEST:
      return
    ser = util_classifier.aggregatePredictions(
        DF_PRED, threshold=0.8)
    self.assertTrue([i == v for i, v in ser.items()])

  def testMakeFStatSer(self):
    # TESTING
    ser = util_classifier.makeFStatSer(self.df_X_short,
        self.ser_y_short)
    self.assertGreater(len(self.df_X_short.columns),
        len(ser))
    self.assertEqual(sum(ser.isnull()), 0)
    self.assertEqual(sum(ser == np.inf), 0)
    #
    ser = util_classifier.makeFStatSer(self.df_X_long,
        self.ser_y_long)
    self.assertGreater(len(self.df_X_long.columns),
        len(ser))

  def testPlotStateFstat(self):
    raise RuntimeError("Not implemented")


if __name__ == '__main__':
  unittest.main()
