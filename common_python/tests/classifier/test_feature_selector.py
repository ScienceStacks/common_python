import common_python.constants as cn
from common_python.testing import helpers
from common_python.tests.classifier import helpers as test_helpers
from common_python.classifier import feature_selector
from common_python.testing import helpers
from common_python.util.persister import Persister

import copy
import os
import pandas as pd
import numpy as np
from sklearn import svm
import unittest

IGNORE_TEST = False


DIR_PATH = os.path.abspath(os.path.dirname(__file__))
TEST_DATA_PATH = os.path.join(DIR_PATH,
    "test_feature_selector.pcl")
PERSISTER = Persister(TEST_DATA_PATH)

if not PERSISTER.isExist():
  DF_X, SER_Y = test_helpers.getDataLong()
  PERSISTER.set([DF_X, SER_Y])
else:
  try:
    [DF_X, SER_Y] = PERSISTER.get()
  except:
    DATA = None
    DATA_LONG = None



def addTest(instance):
  def test(num_iteration):
    instance._init()
    for cls in instance.ser_y.unique():
      for _ in range(num_iteration):
        instance.selector.add(cls)
      length = len(instance.selector.feature_dct[cls])
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
    ordered_dct = self.selector.ordered_dct
    diff = set(ordered_dct.keys()).symmetric_difference(
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
    CLS = 0
    self.selector.add(CLS)
    self.selector.add(CLS, feature=FEATURE)
    self.assertEqual(self.selector.feature_dct[CLS][-1],
        FEATURE)

  def testZeroValues(self):
    if IGNORE_TEST:
      return
    self._init()
    CLS = 0
    self.selector.add(CLS)
    df_X = self.selector.zeroValues(CLS)
    self.assertTrue(any([v != 0 for v in
        df_X[self.selector.feature_dct[CLS]]]))
    features = set(self.selector.ordered_dct[
        CLS]).difference(self.selector.feature_dct[CLS])
    for feature in features:
      self.assertTrue(all(
          [v == 0 for v in df_X[feature]]))

  def testRemove(self):
    if IGNORE_TEST:
      return
    CLS = 0
    self.selector.add(CLS)
    self.assertEqual(len(self.selector.feature_dct[CLS]), 1)
    self.selector.add(CLS)
    self.assertEqual(len(self.selector.feature_dct[CLS]),
        2)
    last_feature = self.selector.feature_dct[CLS][-1]
    self.selector.remove(CLS)
    self.assertEqual(len(self.selector.feature_dct[CLS]),
        1)
    self.selector.add(CLS)
    self.assertNotEqual(last_feature,
        self.selector.feature_dct[CLS][-1])

  def testRemoveSpecifiedFeature(self):
    if IGNORE_TEST:
      return
    self._init()
    CLS = 0
    self.selector.add(CLS)
    self.selector.add(CLS)
    feature_remove = self.selector.feature_dct[CLS][0]
    feature_stay = self.selector.feature_dct[CLS][1]
    self.selector.remove(CLS, feature=feature_remove)
    self.assertEqual(len(self.selector.feature_dct[CLS]),
        1)
    self.assertEqual(self.selector.feature_dct[CLS][0],
        feature_stay)
    self.assertEqual(self.selector.remove_dct[CLS][0],
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
      for cls in self.ser_y.unique():
        for _ in range(num_iteration):
          if not selector.add(cls):
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
