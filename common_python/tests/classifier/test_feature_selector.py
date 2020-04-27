import common_python.constants as cn
from common_python.testing import helpers
from common_python.tests.classifier import helpers as test_helpers
from common_python.classifier import feature_selector
from common_python.testing import helpers

import copy
import os
import pandas as pd
import numpy as np
from sklearn import svm
import unittest

IGNORE_TEST = False
CLASS = 1

DF_X, SER_Y_ALL = test_helpers.getDataLong()
SER_Y = pd.Series([
    binary_feature_manager.PCLASS if v == CLASS
    else binary_feature_manager.NCLASS
    for v in SER_Y_ALL],
    index=SER_Y_ALL.index)


def addTest(instance):
  def test(num_iteration):
    instance._init()
    for _ in range(num_iteration):
      instance.selector.add()
    length = len(instance.selector.features)
    if length != num_iteration:
      self.assertEqual(length, num_iteration)
  #
  test(1)
  test(5)


##################################################
class TestFeatureSelector(unittest.TestCase):
  
  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()
  
  def _init(self):
    self.df_X, self.ser_y = (copy.deepcopy(DF_X),
        copy.deepcopy(SER_Y))
    self.selector = feature_selector.FeatureSelector(
      DF_X, SER_Y)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertTrue(self.df_X.equals(
        self.selector._df_X))
    #
    all_features = self.selector.all
    diff = set(all_features).symmetric_difference(
        self.ser_y.unique())
    self.assertEqual(len(diff), 0)

  def testAdd(self):
    if IGNORE_TEST:
      return
    addTest(self)

  def testAddSpecific(self):
    if IGNORE_TEST:
      return
    self._init()
    FEATURE = "DUMMY"
    self.selector.add()
    self.selector.add(feature=FEATURE)
    self.assertEqual(self.selector.features[-1],
        FEATURE)

  def testRemove(self):
    if IGNORE_TEST:
      return
    self.selector.add()
    self.assertEqual(len(self.selector.features), 1)
    self.selector.add()
    self.assertEqual(len(self.selector.features),
        2)
    last_feature = self.selector.features[-1]
    self.selector.remove()
    self.assertEqual(len(self.selector.features),
        1)
    self.selector.add()
    self.assertNotEqual(last_feature,
        self.selector.features[-1])

  def testRemoveSpecifiedFeature(self):
    if IGNORE_TEST:
      return
    self._init()
    CLS = 0
    self.selector.add()
    self.selector.add()
    feature_remove = self.selector.features[0]
    feature_stay = self.selector.features[1]
    self.selector.remove(CLS, feature=feature_remove)
    self.assertEqual(len(self.selector.features),
        1)
    self.assertEqual(self.selector.features[0],
        feature_stay)
    self.assertEqual(self.selector.removes[0],
        feature_remove)


##################################################
class TestFeatureSelectorCorr(unittest.TestCase):
  
  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()
  
  def _init(self):
    self.df_X, self.ser_y = (copy.deepcopy(DF_X),
        copy.deepcopy(SER_Y))
    self.selector = feature_selector.FeatureSelectorCorr(
      DF_X, SER_Y)

  def testAdd(self):
    if IGNORE_TEST:
      return
    def test(num_iteration, max_corr, expected_size):
      selector = feature_selector.FeatureSelectorCorr(
          self.df_X, self.ser_y, max_corr=max_corr)
      for _ in range(num_iteration):
        if not selector.add():
          break
      self.assertEqual(len(selector.feature_dct[cls]),
          expected_size)
    #
    test(5, 0, 1)
    test(5, 1, 5)


##################################################
class TestFeatureSelectorResidual(unittest.TestCase):
  
  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()
  
  def _init(self):
    self.df_X, self.ser_y = (copy.deepcopy(DF_X),
        copy.deepcopy(SER_Y))
    self.selector \
        = feature_selector.FeatureSelectorResidual(
        DF_X, SER_Y)

  def testAdd(self):
    if IGNORE_TEST:
      return
    addTest(self)
    

if __name__ == '__main__':
  unittest.main()
