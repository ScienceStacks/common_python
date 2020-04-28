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
    cn.PCLASS if v == CLASS else cn.NCLASS
    for v in SER_Y_ALL], index=SER_Y_ALL.index)


def addTest(instance):
  def test(num_iteration):
    instance._init()
    for _ in range(num_iteration):
      instance.selector.add()
    length = len(instance.selector.chosens)
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
    self._init()
    self.assertTrue(self.df_X.equals(
        self.selector._df_X))
    #
    diff = set(self.selector.all).symmetric_difference(
        self.df_X.columns.tolist())
    self.assertEqual(len(diff), 0)

  def testAdd(self):
    if IGNORE_TEST:
      return
    addTest(self)

  def testAddSpecific(self):
    if IGNORE_TEST:
      return
    FEATURE = "DUMMY"
    self.selector.add()
    self.selector.add(feature=FEATURE)
    self.assertEqual(self.selector.chosens[-1],
        FEATURE)

  def testRemove(self):
    if IGNORE_TEST:
      return
    self._init()
    self.selector.add()
    self.assertEqual(len(self.selector.chosens), 1)
    self.selector.add()
    self.assertEqual(len(self.selector.chosens),
        2)
    last_feature = self.selector.chosens[-1]
    self.selector.remove()
    self.assertEqual(len(self.selector.chosens),
        1)
    self.selector.add()
    self.assertNotEqual(last_feature,
        self.selector.chosens[-1])

  def testRemoveSpecifiedFeature(self):
    if IGNORE_TEST:
      return
    self.selector.add()
    self.selector.add()
    feature_remove = self.selector.chosens[0]
    feature_stay = self.selector.chosens[1]
    self.selector.remove(feature=feature_remove)
    self.assertEqual(len(self.selector.chosens),
        1)
    self.assertEqual(self.selector.chosens[0],
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
    addTest(self)


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
    self._init()
    addTest(self)
    

if __name__ == '__main__':
  unittest.main()
