import common_python.constants as cn
from common_python.testing import helpers
from common_python.tests.classifier import helpers as test_helpers
from common_python.classifier import multi_classifier
from common_python.testing import helpers

import pandas as pd
import numpy as np
from sklearn import svm
import unittest

IGNORE_TEST = False


class TestFeatureHandler(unittest.TestCase):
  
  def setUp(self):
    self.df_X, self.ser_y = test_helpers.getDataLong()
    self.handler = multi_classifier.FeatureHandler(
        self.df_X, self.ser_y)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertTrue(self.df_X.equals(
        self.handler.df_X))

  def testFeatureDct(self):
    if IGNORE_TEST:
      return
    feature_dct = self.handler.feature_dct
    diff = set(feature_dct.keys()).symmetric_difference(
        self.ser_y.unique())
    self.assertEqual(len(diff), 0)

  def testGetFeatures(self):
    if IGNORE_TEST:
      return
    for cls in self.ser_y.index:
      features = self.handler.getFeatures(cls)
      diff = set(features).symmetric_difference(
          self.df_X.columns)
      self.assertEqual(len(diff), 0)

  def testGetNonFeatures(self):
    if IGNORE_TEST:
      return
    SIZE = 10
    for cls in self.ser_y.unique():
      features = self.handler.getFeatures(cls)
      non_features = self.handler.getNonFeatures(
          features[:SIZE])
      diff = set(non_features).symmetric_difference(
          features[:SIZE])
      self.assertEqual(len(diff), len(self.df_X.columns))


class TestFeatureHandler(unittest.TestCase):
  
  def setUp(self):
    self.df_X, self.ser_y = test_helpers.getDataLong()
    self.clf = multi_classifier.MultiClassifier()

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertTrue(isinstance(self.clf.base_clf,
        svm.LinearSVC))

  def testFit(self):
    if IGNORE_TEST:
      return
    self.clf.fit(self.df_X, self.ser_y)
    import pdb; pdb.set_trace()
    

if __name__ == '__main__':
  unittest.main()


