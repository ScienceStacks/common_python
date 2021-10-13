"""Tests for classifier_ensemble.ClassifierEnsemble."""

import common_python.constants as cn
from common_python.classifier import classifier_collection
from common_python.classifier import classifier_ensemble
from common_python.classifier import feature_analyzer
from common_python.classifier.classifier_ensemble  \
    import ClassifierEnsemble, ClassifierDescriptorSVM
from common_python.classifier.classifier_collection  \
    import ClassifierCollection
from common_python.testing import helpers
from common_python.util import util
from common_python.tests.classifier import helpers as test_helpers
from common.trinary_data import TrinaryData

import collections
import copy
import os
import pandas as pd
import random
from sklearn import svm
import numpy as np
import unittest

IGNORE_TEST = True
IS_PLOT = True
SIZE = 10
ITERATIONS = 3
values = list(range(SIZE))
values.extend(values)
DF = pd.DataFrame({
    'A': [10*v for v in values],
    'B': np.repeat(1, 2*SIZE),
    })
SER = pd.Series(values)
TEST_FILE = "test_classifier_ensemble.pcl"
REPO_PATH = util.findRepositoryRoot("xstate")
DATA_PATH = os.path.join(REPO_PATH, "data")
ANALYZER_PATH_REPL = os.path.join(DATA_PATH, "feature_analyzer_with_replicas")
ANALYZER_PATH_REPL_PAT = os.path.join(ANALYZER_PATH_REPL, "%d") 
ANALYZER_REPL_DCT = feature_analyzer.deserialize(
    {s: ANALYZER_PATH_REPL_PAT % s for s in range(6)})
ANALYZER_PATH_AVG = os.path.join(DATA_PATH, "feature_analyzer_averaged")
ANALYZER_PATH_AVG_PAT = os.path.join(ANALYZER_PATH_AVG, "%d") 
ANALYZER_AVG_DCT = feature_analyzer.deserialize(
    {s: ANALYZER_PATH_AVG_PAT % s for s in range(5)})
DF_X, SER_Y = test_helpers.getData()
DF_X_LONG, SER_Y_LONG = test_helpers.getDataLong()
HOLDOUTS = 1
SVM_ENSEMBLE = ClassifierEnsemble( ClassifierDescriptorSVM(),
    filter_high_rank=10,
    size=SIZE, holdouts=HOLDOUTS)
FITTED_SVM_ENSEMBLE_LONG = copy.deepcopy(SVM_ENSEMBLE)
FITTED_SVM_ENSEMBLE_LONG.fit(DF_X_LONG, SER_Y_LONG)
CLASSES = list(set(SER_Y.values))
CLASSES_LONG = list(set(SER_Y_LONG.values))


######## Helper Classes ######
class RandomClassifier(object):

  def __init__(self):
    self.class_probs = None
    self.coef_ = []
    if os.path.exists(TEST_FILE):
      os.remove(TEST_FILE)

  def fit(self, df_X, ser_y):
    """
    Fits by recording probability of each class.
    :param object _:
    :param pd.Series ser_y:
    """
    class_counts = dict(collections.Counter(ser_y.values.tolist()))
    self.class_probs = {k: v / len(ser_y) 
         for k,v in class_counts.items()}
    self.coef_ = np.repeat(0.1, len(df_X.columns))

  def predict(self, df_X):
    """
    :param pd.DataFrame df_X:
    :return pd.Series:
    """
    randoms = np.random.randint(0, len(self.class_probs.keys()),
        len(df_X))
    values = [list(self.class_probs.keys())[n] for n in randoms]
    ser = pd.Series(values, index=df_X.index)
    return ser

  def score(self, df_X, ser_y):
    ser = self.predict(df_X)
    value = np.mean(ser == ser_y)
    return value


class ClassifierDescriptorRandom(
    classifier_ensemble.ClassifierDescriptor):
  # Descriptor information needed for Random classifiers
  # Descriptor is for one-vs-rest. So, there is a separate
  # classifier for each class.
  
  def __init__(self, clf=RandomClassifier()):
    self.clf = clf

  def getImportance(self, clf, class_selection=None):
    """
    Calculates the importances of features.
    :param Classifier clf:
    :param int class_selection: class for which importance is computed
    :return list-float:
    """
    return self.clf.coef_

RANDOM_ENSEMBLE = ClassifierEnsemble(
        ClassifierDescriptorRandom(),
        size=SIZE, holdouts=HOLDOUTS)
RANDOM_ENSEMBLE.fit(DF, SER)


######## Test Classes ######
class TestClassifierEnsemble(unittest.TestCase):

  def _init(self):
    self.df_X_long = copy.deepcopy(DF_X_LONG)
    self.df_X = copy.deepcopy(DF_X)
    self.ser_y_long = copy.deepcopy(SER_Y_LONG)
    self.ser_y = copy.deepcopy(SER_Y)
    self.svm_ensemble = copy.deepcopy(SVM_ENSEMBLE)
    self.classifier_ensemble_random = copy.deepcopy(RANDOM_ENSEMBLE)
  
  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(len(self.svm_ensemble.clfs), 0)
    self.assertEqual(len(self.svm_ensemble.features), 0)
    self.assertEqual(len(self.svm_ensemble.classes), 0)
    self.assertTrue(isinstance(self.svm_ensemble.clf_desc.clf,
        svm.LinearSVC))

  def testFit(self):
    if IGNORE_TEST:
      return
    self._init()
    holdouts = 1
    #
    def test(filter_high_rank):
      if filter_high_rank is None:
        filter_high_rank = len(self.df_X.columns)
      svm_ensemble = ClassifierEnsemble(ClassifierDescriptorSVM(),
        size=SIZE, holdouts=holdouts,
        filter_high_rank=filter_high_rank)
      svm_ensemble.fit(self.df_X, self.ser_y)
      for items in [svm_ensemble.clfs, svm_ensemble.scores]:
        self.assertEqual(len(items), SIZE)
      self.assertEqual(len(svm_ensemble.features), filter_high_rank)
    #
    test(10)
    test(None)

  def testOrderFeatures(self):
    if IGNORE_TEST:
      return
    self._init()
    self.svm_ensemble.fit(self.df_X, self.ser_y)
    clf = self.svm_ensemble.clfs[0]
    #
    def test(class_selection):
       result = self.svm_ensemble._orderFeatures(clf,
           class_selection=class_selection)
       self.assertEqual(len(result), len(self.svm_ensemble.features))
       trues = [ v in range(1, len(result)+1) for v in result]
       self.assertTrue(all(trues))
    #
    test(None)
    test(1)

  def testMakeRankDF(self):
    if IGNORE_TEST:
      return
    self.svm_ensemble.fit(self.df_X, self.ser_y)
    df = self.svm_ensemble.makeRankDF()
    self.assertTrue(helpers.isValidDataFrame(df,
        [cn.MEAN, cn.STD, cn.STERR]))

  def testMakeImportanceDF(self):
    if IGNORE_TEST:
      return
    self.svm_ensemble.fit(self.df_X, self.ser_y)
    #
    def test(class_selection):
      df = self.svm_ensemble.makeImportanceDF(
          class_selection=class_selection)
      self.assertTrue(helpers.isValidDataFrame(df,
          [cn.MEAN, cn.STD, cn.STERR]))
      trues = [df.loc[df.index[i], cn.MEAN]
           >= df.loc[df.index[i+1], cn.MEAN] 
          for i in range(len(df.index)-1)]
    #
    test(None)
    test(1)

  def testPlotImportance(self):
    if IGNORE_TEST:
      return
    self.svm_ensemble.fit(self.df_X, self.ser_y)
    # Smoke tests
    _ = self.svm_ensemble.plotImportance(top=40, title="SVM",
        is_plot=IS_PLOT, ylabel="XXX")
    _ = self.svm_ensemble.plotImportance(top=40, title="SVM-class 2", 
        class_selection=2, is_plot=IS_PLOT)

  def testGetFeatureContributions(self):
    if IGNORE_TEST:
      return
    self._init()
    self.svm_ensemble.fit(self.df_X_long, self.ser_y_long)
    clf = self.svm_ensemble.clfs[0]
    df = self.svm_ensemble.clf_desc.getFeatureContributions(clf,
        self.svm_ensemble.columns, self.df_X_long.loc["T1.0", :])
    self.assertEqual(len(df), len(CLASSES_LONG))
    self.assertTrue(isinstance(df, pd.DataFrame))
    diff = set(self.svm_ensemble.columns).symmetric_difference(df.columns)
    self.assertEqual(len(diff), 0)

  def testPlotFeatureContributions(self):
    # TESTING
    self._init()
    instance = "T3.0"
    svm_ensemble = copy.deepcopy(FITTED_SVM_ENSEMBLE_LONG)
    ser_X = self.df_X_long.loc[instance]
    svm_ensemble.plotFeatureContributions(ser_X, is_plot=IS_PLOT,
        title=instance, true_class=self.ser_y_long.loc[instance])

  def testPlotRank(self):
    if IGNORE_TEST:
      return
    self.svm_ensemble.fit(self.df_X, self.ser_y)
   # Smoke tests
    _ = self.svm_ensemble.plotRank(top=40, title="SVM", is_plot=IS_PLOT)

  def testCrossValidate1(self):
    # Tests the cross validation of ClassifierEnsemble
    if IGNORE_TEST:
      return
    self._init()
    cvrs = []
    num_clfs = 10 # number of classifiers created randomly
    for fltr in [6, 1515]:
      clf = classifier_ensemble.ClassifierEnsemble(
          classifier_ensemble.ClassifierDescriptorSVM(),
          filter_high_rank=fltr, size=10)
      cvrs.append(
          classifier_collection.ClassifierCollection.crossValidateByState(
          clf, self.df_X, self.ser_y, num_clfs))
    self.assertGreater(cvrs[1], cvrs[0])

  def testRandomClassifier(self):
    if IGNORE_TEST:
      return
    clf = RandomClassifier()
    clf.fit(DF, SER)
    self.assertEqual(clf.class_probs[0], 1/SIZE)
    ser_predict = clf.predict(DF)
    trues = [c in range(SIZE) for c in ser_predict.values]
    self.assertTrue(all(trues))

  def testPredict(self):
    if IGNORE_TEST:
      return
    self._init()
    ser = self.classifier_ensemble_random.predict(DF)
    mean = ser.mean(axis=1).values[0]
    expected = 1/SIZE
    self.assertLess(abs(mean - expected), 0.1)
# TODO: Create tests that use RandomClassifier or delete this

  def testScore(self):
    if IGNORE_TEST:
      return
    self._init()
    score = self.classifier_ensemble_random.score(DF, SER)
    expected = 1/SIZE
    self.assertLess(abs(score- expected), 0.1)

  def testSerializeAndDeserialize(self):
    if IGNORE_TEST:
      return
    self.svm_ensemble.fit(self.df_X, self.ser_y)
    self.svm_ensemble.serialize(TEST_FILE)
    svm_ensemble = ClassifierEnsemble.deserialize(TEST_FILE)
    diff = set(self.svm_ensemble.features).symmetric_difference(
        svm_ensemble.features)
    self.assertEqual(len(diff), 0)

  def testMakeInstancePredictionDF(self):
    if IGNORE_TEST:
      return
    self._init()
    ser = self.svm_ensemble.makeInstancePredictionDF(
        self.df_X, self.ser_y)
    num = sum([v1 == v2 for v1, v2 in 
        zip(self.ser_y.values, ser.values)])
    self.assertGreater(num/len(ser), 0.9)
    #
    self.svm_ensemble = ClassifierEnsemble(
        ClassifierDescriptorSVM(),
        size=SIZE, holdouts=1)

  def testCalcAdjProbTail(self):
    if IGNORE_TEST:
      return
    def test(prob):
      self.assertTrue(isinstance(prob, float))
      self.assertGreaterEqual(prob, 0)
      self.assertLessEqual(prob, 1)
    self._init()
    prob_long =  self.svm_ensemble.calcAdjStateProbTail(
        self.df_X_long, self.ser_y_long)
    test(prob_long)
    prob_short =  self.svm_ensemble.calcAdjStateProbTail(
        self.df_X, self.ser_y)
    test(prob_short)
    self.assertGreater(prob_short, prob_long)

  def testCrossValidate2(self):
    if IGNORE_TEST:
      return
    self._init()
    accuracies = []
    for rank in [1, 3, 15]:
      accuracy = self.svm_ensemble.crossValidate(self, num_holdout=1,
          num_iter=10, filter_high_rank=rank)
      accuracies.append(accuracy)
    for idx in range(len(accuracies) - 1):
      self.assertLessEqual(accuracies[idx], accuracies[idx+1])

  def testPoorPerformanceIndidualReplicas(self):
    if IGNORE_TEST:
      return
    # Using individual replicas is producing bad classifier performance
    IDXs = ["T3.0", "T3.1", "T3.2"]
    class ResultAnalysis():
      def __init__(self, trinary, path, predictions=None):
        self.trinary = trinary
        self.predictions = predictions
        self.df_prediction = None
        self.predictions = None
        self.ensemble = None
    # Cases to analyze
    analysis_dct = {}
    analysis_dct["repl-regl"] = ResultAnalysis(TrinaryData(
        is_averaged=False, is_dropT1=False, is_regulator=True),
        ANALYZER_PATH_REPL)
    analysis_dct["repl-notregl"] = ResultAnalysis(TrinaryData(
        is_averaged=False, is_dropT1=False, is_regulator=False),
        ANALYZER_PATH_REPL)
    analysis_dct["avg-regl"] = ResultAnalysis(TrinaryData(
       is_averaged=True, is_dropT1=False, is_regulator=True),
        ANALYZER_PATH_AVG)
    analysis_dct["avg-notregl"] = ResultAnalysis(TrinaryData(
        is_averaged=True, is_dropT1=False, is_regulator=False),
        ANALYZER_PATH_AVG)
    #
    def evaluate(idxs):
      """
      Evaluates predictive performance. Updates the ResultAnalysis
      objects in analysis_dct.
 
      Parameters
      ----------
      idxs: list-str
      """
      for key, result_analysis in analysis_dct.items():
        if "avg-" in key:
          new_idxs = list(set([i[0:-2] for i in idxs]))
        else:
          new_idxs = idxs
        # 
        df_X = result_analysis.trinary.df_X
        ser_y = result_analysis.trinary.ser_y
        states = list(set(ser_y.values))
        #
        ensemble = classifier_ensemble.ClassifierEnsemble(
            filter_high_rank=100, size=100)
        ensemble.fit(df_X, ser_y)
        df_X_test = df_X.loc[new_idxs, :]
        ser_y_test = ser_y.loc[new_idxs]
        df_predict = ensemble.predict(df_X_test)
        df_predict["true state"] = ser_y_test
        # Construct the predictions
        predictions = []
        for clf in ensemble.clfs:
          df_X_test_sub = df_X[ensemble.columns]
          dct = {i: [] for i in states}
          for idx in new_idxs:
            {i: dct[i].append(clf.coef_[i].dot(df_X_test_sub.loc[idx, :]))
                for i in states}
          df_result = pd.DataFrame(dct, index=new_idxs)
          predictions.append(df_result)
        result_analysis.df_predict = df_predict
        result_analysis.predictions = predictions
        result_analysis.ensemble = ensemble
    #
    evaluate(IDXs) # Updates objects in analysis_dct
     

if __name__ == '__main__':
  unittest.main()
