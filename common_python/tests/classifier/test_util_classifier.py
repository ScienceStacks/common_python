from common_python import constants as cn
from common_python.classifier import util_classifier
from common_python.util import util
from common_python.tests.classifier  \
    import helpers as test_helpers
from common_python.classifier import classifier_ensemble
from common_python.testing import helpers

import copy
import itertools
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
DF_X_BINARY = DF_X_BINARY.sort_index()
FEATURES = DF_X_BINARY.columns.tolist()
SER_Y = SER_Y.sort_index()
CLASS = 1
SER_Y_BINARY = pd.Series([
    cn.PCLASS if v == CLASS else cn.NCLASS
    for v in SER_Y_ALL],
    index=SER_Y_ALL.index)
CLF = svm.LinearSVC()

################ TESTS #################
class TestFunctions(unittest.TestCase):

  def setUp(self):
    data = test_helpers.getData()
    self.df_X_short, self.ser_y_short = data
    self.df_X_long, self.ser_y_long =  \
        test_helpers.getDataLong()
    self.clf = copy.deepcopy(CLF)

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
    # Test on all samples with a subset of features
    features = random.sample(all_features, SIZE)
    score = util_classifier.scoreFeatures(
        CLF, DF_X_BINARY, SER_Y_BINARY,
        features=features)
    self.assertGreater(score_all, score)
    # Train on all features with a subset of samples
    for _ in range(5):
      # Ensure that sample has 2 classes
      train_idxs = random.sample(
          DF_X_BINARY.index.tolist(), SIZE)
      classes = SER_Y_BINARY.loc[train_idxs].unique()
      if len(classes) > 1:
        break
    score = util_classifier.scoreFeatures(
        CLF, DF_X_BINARY, SER_Y_BINARY,
        train_idxs=train_idxs)
    self.assertGreater(score_all, score)
    # Test on a subset of samples
    test_idxs = SER_Y_BINARY[
        SER_Y_BINARY==cn.PCLASS].index.tolist()
    test_idxs.extend(random.sample(
        DF_X_BINARY.index.tolist(), len(test_idxs)))
    score = util_classifier.scoreFeatures(
        CLF, DF_X_BINARY, SER_Y_BINARY,
        test_idxs=test_idxs)
    self.assertTrue(np.isclose(score_all, score))

  def testPartitionByState(self):
    if IGNORE_TEST:
      return
    train_idxs, test_idxs =  \
        util_classifier.partitionByState(SER_Y,
        holdouts=1)
    # all classes are present?
    classes = SER_Y.unique()
    test_classes = SER_Y.loc[test_idxs]
    diff = set(classes).symmetric_difference(
        test_classes)
    self.assertEqual(len(diff), 0)
    # Cover all indices in the data?
    all_indices = set(train_idxs).union(test_idxs)
    diff = set(SER_Y.index).symmetric_difference(
        all_indices)
    self.assertEqual(len(diff), 0)
    # Successive calls yield different indices?
    train_idxs2, test_idxs2 =  \
        util_classifier.partitionByState(SER_Y,
        holdouts=1)
    diff = set(train_idxs).symmetric_difference(
        train_idxs2)
    self.assertGreater(len(diff), 0)

  def testBinaryCrossValidate(self):
    if IGNORE_TEST:
      return
    SIZE = 20
    partitions = [util_classifier.partitionByState(
        SER_Y_BINARY, holdouts=1)
        for _ in range(SIZE)]
    partitioner = util_classifier.partitioner(
        SER_Y_BINARY, SIZE, num_holdout=1)
    score_one = util_classifier.scoreFeatures(
        CLF, DF_X_BINARY, SER_Y_BINARY)
    bcv_result = util_classifier.binaryCrossValidate(CLF,
        DF_X_BINARY, SER_Y_BINARY,
        partitions=partitioner)
    score_many = bcv_result.score
    self.assertGreaterEqual(score_one, score_many)
    #
    bcv_result = util_classifier.binaryCrossValidate(CLF,
        DF_X_BINARY, SER_Y_BINARY, num_holdouts=1,
        num_iteration=SIZE)
    score_many2 = bcv_result.score
    self.assertEqual(len(bcv_result.clfs), SIZE)
    self.assertLess(np.abs(score_many-score_many2),
        0.1)

  def testBinaryMultiFit(self):
    if IGNORE_TEST:
      return
    SIZE = 20
    FEATURES = ["Rv2009", "Rv3830c"]
    partitions = [util_classifier.partitionByState(
        SER_Y_BINARY, holdouts=1)
        for _ in range(SIZE)]
    fit_partitions = [t for t,_ in partitions]
    bcv_result = util_classifier.binaryCrossValidate(CLF,
        DF_X_BINARY[FEATURES], SER_Y_BINARY,
        partitions=partitions)
    intercept, coefs = util_classifier.binaryMultiFit(CLF,
        DF_X_BINARY[FEATURES], SER_Y_BINARY,
        list_train_idxs=fit_partitions)
    values1 = [intercept]
    values1.extend(coefs.tolist())
    values2 = [np.mean([c.intercept_[0] 
         for c in bcv_result.clfs])]
    mean_coefs = []
    [mean_coefs.extend(c.coef_) for c in bcv_result.clfs]
    mean_coefs = sum(mean_coefs)/SIZE
    values2.extend(mean_coefs.tolist())
    # Check that the two results are equivalent in that
    # for every assignment of trainary value, we have
    # the same trinary value
    trinary_values = [-1, 0, 1]
    iterator = itertools.product(trinary_values,
        trinary_values)
    for f1, f2 in iterator:
      vec = np.array([1, f1, f2])
      sum1 = util.makeTrinary(sum(vec*values1))
      sum2 = util.makeTrinary(sum(vec*values2))
      self.assertEqual(sum1, sum2)

  def testCorrelatePrediction(self):
    if IGNORE_TEST:
      return
    clf2 = copy.deepcopy(CLF)
    NUM_PARTITIONS = 10
    FEATURE = "Rv0081"
    partitions = [util_classifier.partitionByState(
        SER_Y_BINARY) for _ in range(NUM_PARTITIONS)]
    clf_desc1 = util_classifier.ClassifierDescription(
        clf=self.clf, features=FEATURE)
    clf_desc2 = util_classifier.ClassifierDescription(
        clf=self.clf, features=FEATURE)
    score = util_classifier.correlatePredictions(
        clf_desc1, clf_desc2, DF_X_BINARY,
        SER_Y_BINARY, partitions)
    self.assertGreater(score, 0.98)
    #
    clf_desc2 = util_classifier.ClassifierDescription(
        clf=clf2, features=FEATURES[1])
    score = util_classifier.correlatePredictions(
        clf_desc1, clf_desc2, DF_X_BINARY,
        SER_Y_BINARY, partitions)
    self.assertTrue(isinstance(score, float))

  def testPartitioner(self):
    if IGNORE_TEST:
      return
    SIZE = 3
    COUNT = 4
    SER = pd.Series({v: v % SIZE 
        for v in range(SIZE*COUNT)})
    iterator = util_classifier.partitioner(SER, COUNT)
    count = 0
    for train_set, test_set in iterator:
      all = set(train_set).union(test_set)
      diff = all.symmetric_difference(set(SER.index))
      self.assertEqual(len(diff), 0)
      count += 1
    self.assertEqual(COUNT, count)

  def testMakePartitioner(self):
    if IGNORE_TEST:
      return
    def isEquiv(set1, set2):
      self.assertEqual(len(set1), len(set2))
    #
    def test(partition1, partition2):
      self.assertEqual(len(partition1), len(partition2))
      for idx in range(len(partition1)):
        for _ in range(2):
          isEquiv(partition1[idx][0], partition2[idx][0])
          isEquiv(partition1[idx][1], partition2[idx][1])
    #
    SIZE = 3
    COUNT = 4
    NUM_HOLDOUT = 2
    SER = pd.Series({v: v % SIZE 
        for v in range(SIZE*COUNT)})
    ref_iter = util_classifier.partitioner(SER, COUNT,
        num_holdout=NUM_HOLDOUT)
    ref_partition = [(p, q) for p,q in ref_iter]
    #
    iterator = util_classifier.makePartitioner(
        ser_y=SER, 
        num_iteration=COUNT, num_holdout=NUM_HOLDOUT)
    this_partition = [(p, q) for p,q in iterator]
    test(ref_partition, this_partition)
    #
    iterator = util_classifier.makePartitioner(
        partitions=this_partition)
    this_partition = [(p, q) for p,q in iterator]
    test(ref_partition, this_partition)

  def testBackEliminate(self):
    if IGNORE_TEST:
      return
    BASE_FEATURES = ["Rv3095", "Rv3246c"]
    BOGUS_FEATURES = ["Rv0054"]
    FEATURES = list(set(BASE_FEATURES).union(
        BOGUS_FEATURES))
    #
    df_X = DF_X_BINARY[FEATURES]
    partitions = util_classifier.makePartitions(
        ser_y=SER_Y_BINARY, num_iteration=10)
    bcv_result = util_classifier.binaryCrossValidate(CLF,
        df_X, SER_Y_BINARY, partitions)
    base_score = bcv_result.score
    #
    def test(features, score):
        if len(features) > 1:
          # Occassionally we sample indexes for which
          # a single feature is sufficient. Ignore these.
          diff = set(features).symmetric_difference(
              BASE_FEATURES)
          self.assertEqual(len(diff), 0)
        self.assertTrue(np.isclose(base_score, score))
    #
    features, score = util_classifier.backEliminate(
        CLF, df_X, SER_Y_BINARY, partitions)
    test(features, score)
    #
    df_X = DF_X_BINARY[BASE_FEATURES]
    features, score = util_classifier.backEliminate(
        CLF, df_X, SER_Y_BINARY, partitions)
    test(features, score)


if __name__ == '__main__':
  unittest.main()
