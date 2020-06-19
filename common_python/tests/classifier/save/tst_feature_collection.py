import common_python.constants as cn
from common_python.testing import helpers
from common_python.tests.classifier  \
    import helpers as test_helpers
from common_python.classifier import feature_collection
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
      instance.collection.add()
    length = len(instance.collection.chosens)
    if length != num_iteration:
      instance.assertEqual(length, num_iteration)
  #
  test(1)
  test(5)


##################################################
class TestFeatureCollection(unittest.TestCase):

  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()

  def _init(self):
    self.df_X, self.ser_y = (copy.deepcopy(DF_X),
        copy.deepcopy(SER_Y))
    self.collection = feature_collection.FeatureCollection(
      DF_X, SER_Y)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self._init()
    self.assertTrue(self.df_X.equals(
        self.collection._df_X))
    #
    diff = set(self.collection.all).symmetric_difference(
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
    self.collection.add()
    self.collection.add(feature=FEATURE)
    self.assertEqual(self.collection.chosens[-1],
        FEATURE)

  def testRemove(self):
    if IGNORE_TEST:
      return
    self._init()
    self.collection.add()
    self.assertEqual(len(self.collection.chosens), 1)
    self.collection.add()
    self.assertEqual(len(self.collection.chosens),
        2)
    last_feature = self.collection.chosens[-1]
    self.collection.remove()
    self.assertEqual(len(self.collection.chosens),
        1)
    self.collection.add()
    self.assertNotEqual(last_feature,
        self.collection.chosens[-1])

  def testRemoveSpecifiedFeature(self):
    if IGNORE_TEST:
      return
    self.collection.add()
    self.collection.add()
    feature_remove = self.collection.chosens[0]
    feature_stay = self.collection.chosens[1]
    self.collection.remove(feature=feature_remove)
    self.assertEqual(len(self.collection.chosens),
        1)
    self.assertEqual(self.collection.chosens[0],
        feature_stay)
    self.assertEqual(self.collection.removes[0],
        feature_remove)


##################################################
class TestFeatureCollectionCorr(unittest.TestCase):

  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()

  def _init(self):
    self.df_X, self.ser_y = (copy.deepcopy(DF_X),
        copy.deepcopy(SER_Y))
    self.collection = feature_collection.FeatureCollectionCorr(
      DF_X, SER_Y)

  def testAdd(self):
    if IGNORE_TEST:
      return
    addTest(self)


##################################################
class TestFeatureCollectionResidual(unittest.TestCase):

  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()

  def _init(self):
    self.df_X, self.ser_y = (copy.deepcopy(DF_X),
        copy.deepcopy(SER_Y))
    self.collection \
        = feature_collection.FeatureCollectionResidual(
        DF_X, SER_Y)

  def testAdd(self):
    if IGNORE_TEST:
      return
    self._init()
    addTest(self)


if __name__ == '__main__':
  unittest.main()
