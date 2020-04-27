from common_python import constants as cn
from common_python.classifier import util_classifier
from common_python.tests.classifier  \
    import helpers as test_helpers
from common_python.classifier import classifier_ensemble
from common_python.testing import helpers

import pandas as pd
import random
import numpy as np
from sklearn import svm
import unittest


IGNORE_TEST = False
IS_PLOT = False
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
DF_X_BINARY, SER_Y_ALL = test_helpers.getDataLong()
CLASS = 1
SER_Y_BINARY = pd.Series([
    cn.PCLASS if v == CLASS else cn.NCLASS
    for v in SER_Y_ALL],
    index=SER_Y_ALL.index)
CLF = svm.LinearSVC()


class TestFunctions(unittest.TestCase):

  def setUp(self):
    data = test_helpers.getData()
    self.df_X_short, self.ser_y_short = data
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

  def testMakeFstatSer(self):
    if IGNORE_TEST:
      return
    ser = util_classifier.makeFstatSer(self.df_X_short,
        self.ser_y_short)
    self.assertGreater(len(self.df_X_short.columns),
        len(ser))
    self.assertEqual(sum(ser.isnull()), 0)
    self.assertEqual(sum(ser == np.inf), 0)
    #
    ser = util_classifier.makeFstatSer(self.df_X_long,
        self.ser_y_long)
    self.assertGreater(len(self.df_X_long.columns),
        len(ser))

  def testPlotStateFstat(self):
    if IGNORE_TEST:
      return
    # Smoke test
    state = 0
    util_classifier.plotStateFstat(state, self.df_X_long,
        self.ser_y_long, is_plot=IS_PLOT)

  def testPlotInstancePredictions(self):
    if IGNORE_TEST:
      return
    # Smoke test
    svm_ensemble = classifier_ensemble.ClassifierEnsemble(
        classifier_ensemble.ClassifierDescriptorSVM(),
        size=1, filter_high_rank=5)
    ser_pred = svm_ensemble.makeInstancePredictionDF(
        self.df_X_long, self.ser_y_long)
    util_classifier.plotInstancePredictions(
        self.ser_y_long, ser_pred, is_plot=IS_PLOT)

  def testMakeFstatDF(self):
    if IGNORE_TEST:
      return
    df = util_classifier.makeFstatDF(
        self.df_X_long, self.ser_y_long)
    self.assertTrue(helpers.isValidDataFrame(df,
        self.ser_y_long.unique()))
    #
    random_indices = random.sample(
        self.df_X_long.index.tolist(), 10)
    ser_weight = self.ser_y_long.copy()
    ser_weight.loc[:] = 1
    ser_weight.loc[random_indices] = 20
    df2 = util_classifier.makeFstatDF(
        self.df_X_long, self.ser_y_long,
        ser_weight=ser_weight)
    self.assertTrue(helpers.isValidDataFrame(df2,
        self.ser_y_long.unique()))

  def testMakeArrays(self):
    if IGNORE_TEST:
      return
    SIZE = 10
    arr_X, arr_y = util_classifier.makeArrays(
        DF_X_BINARY, SER_Y_BINARY,
        DF_X_BINARY.index[:SIZE])
    self.assertEqual(len(arr_X), SIZE)
    self.assertEqual(len(arr_y), SIZE)

  def testScoreFeatures(self):
    if IGNORE_TEST:
      return
    SIZE = 5
    all_features = DF_X_BINARY.columns.tolist()
    score_all = util_classifier.scoreFeatures(
        CLF, DF_X_BINARY, SER_Y_BINARY)
    self.assertEqual(score_all, 1)
    #
    features = random.sample(all_features, SIZE)
    score = util_classifier.scoreFeatures(
        CLF, DF_X_BINARY, SER_Y_BINARY,
        features=features)
    self.assertGreater(score_all, score)
    #
    train_idxs = SER_Y_BINARY[
        SER_Y_BINARY==cn.PCLASS].index.tolist()
    train_idxs.extend(random.sample(
        DF_X_BINARY.index.tolist(), len(train_idxs)))
    score = util_classifier.scoreFeatures(
        CLF, DF_X_BINARY, SER_Y_BINARY,
        train_idxs=train_idxs)
    self.assertGreater(score_all, score)
    #
    test_idxs = SER_Y_BINARY[
        SER_Y_BINARY==cn.PCLASS].index.tolist()
    test_idxs.extend(random.sample(
        DF_X_BINARY.index.tolist(), len(test_idxs)))
    score = util_classifier.scoreFeatures(
        CLF, DF_X_BINARY, SER_Y_BINARY,
        test_idxs=test_idxs)
    self.assertTrue(np.isclose(score_all, score))
    


if __name__ == '__main__':
  unittest.main()
