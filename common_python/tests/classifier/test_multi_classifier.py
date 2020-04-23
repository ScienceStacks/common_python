import common_python.constants as cn
from common_python.testing import helpers
from common_python.tests.classifier import helpers as test_helpers
from common_python.classifier import multi_classifier
from common_python.classifier import feature_selector
from common_python.testing import helpers
from common_python.util.persister import Persister

import copy
import os
import pandas as pd
import numpy as np
from sklearn import svm
import unittest

IGNORE_TEST = True


DIR_PATH = os.path.abspath(os.path.dirname(__file__))
TEST_DATA_PATH = os.path.join(DIR_PATH,
    "test_multi_classifier.pcl")
PERSISTER = Persister(TEST_DATA_PATH)

if not PERSISTER.isExist():
  DF_X, SER_Y = test_helpers.getDataLong()
  CLF = multi_classifier.MultiClassifier()
  CLF.fit(DF_X, SER_Y)
  PERSISTER.set([DF_X, SER_Y, CLF])
else:
  try:
    [DF_X, SER_Y, CLF] = PERSISTER.get()
  except:
    DATA = None
    DATA_LONG = None


class TestMultiClass(unittest.TestCase):
  
  def setUp(self):
    self.df_X, self.ser_y = DF_X, SER_Y
    self.clf = multi_classifier.MultiClassifier()
    self.clf_fitted = copy.deepcopy(CLF)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertTrue(isinstance(self.clf._base_clf,
        svm.LinearSVC))

  def testFit(self):
    # TESTING
    for fs in [
        feature_selector.FeatureSelector,
        feature_selector.FeatureSelectorCorr,
        feature_selector.FeatureSelectorResidual,
        ]:
      clf = multi_classifier.MultiClassifier(max_iter=2,
          max_degrade=0.4,
          feature_selector_cls=fs)
      clf.fit(self.df_X, self.ser_y)
      trues = [s >= 0.5 for s in clf.score_dct.values()]
      self.assertTrue(all(trues))
      trues = [len(f) > 0 for f 
          in clf.selector.feature_dct.values()]
      self.assertTrue(all(trues))
      trues = [clf.best_score_dct[c] >=
          clf.score_dct[c] for c in clf.classes]
      self.assertTrue(all(trues))

  def testPredict(self):
    if IGNORE_TEST:
      return
    df_pred = self.clf_fitted.predict(self.df_X)
    self.assertTrue(helpers.isValidDataFrame(df_pred,
        expected_columns=self.ser_y.unique()))
    

if __name__ == '__main__':
  unittest.main()


