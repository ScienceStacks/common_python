
from common_python.tests.classifier import helpers as test_helpers
from common_python.classifier.feature_set_collection  \
    import FeatureSetCollection, FeatureSet
from common_python.classifier  \
    import feature_set_collection
from common_python.testing import helpers
from common_python import constants as cn

import numpy as np
import os
import shutil
import unittest

IGNORE_TEST = False
IS_PLOT = False
CLASS = 1
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR_PATH = os.path.join(TEST_DIR,
    "test_feature_set_collection_%d" % CLASS)
TEST_SERIALIZE_DIR = os.path.join(TEST_DIR,
    "test_feature_set_collection_serialize")
ANALYZER = test_helpers.getFeatureAnalyzer()
FEATURE1 = "feature1"
FEATURE2 = "feature2"
MIN_SCORE = 0.9


class TestFeatureSet(unittest.TestCase):

  def _remove(self):
    paths = [os.path.join(TEST_SERIALIZE_DIR,
        feature_set_collection.MISC_PCL)]
    csv_file_names = [feature_set_collection.SER_COMB,
        feature_set_collection.SER_SBFSET]
    [paths.append(
        os.path.join(TEST_SERIALIZE_DIR, "%s.csv" % f))
        for f in csv_file_names]
    for path in paths:
      if os.path.isfile(path):
        os.remove(path)

  def setUp(self):
    self._remove()
    self.fset = FeatureSet([FEATURE1, FEATURE2])

  def tearDown(self):
    self._remove()

  def testStr(self):
    if IGNORE_TEST:
      return
    self.assertEqual(self.fset.str, str(self.fset))

  def testEquals(self):
    if IGNORE_TEST:
      return
    fset = FeatureSet([FEATURE2, FEATURE1])
    self.assertTrue(self.fset.equals(fset))


##########################################
class TestFeatureSetCollection(unittest.TestCase):

  def setUp(self):
    self.collection = FeatureSetCollection(ANALYZER,
        min_score=MIN_SCORE)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertGreater(len(self.collection.ser_sbfset), 0)
    falses = [np.isnan(v) for v in
        self.collection.ser_sbfset]
    self.assertFalse(any(falses))

  def testDisjointify(self):
    if IGNORE_TEST:
      return
    ser = self.collection.disjointify()
    self.assertGreater(len(self.collection.ser_sbfset),
        len(ser))
    ser_fset = feature_set_collection.disjointify(
        self.collection.ser_sbfset, min_score=0.9)
    fset_stgs = self.collection.ser_sbfset.index.tolist()
    # Fails because analyzer.ser_fset is bad
    result = set(fset_stgs).issuperset(
        ser_fset.index.tolist())
    self.assertTrue(result)

  def test_ser_comb(self):
    if IGNORE_TEST:
      return
    ser = self.collection.ser_comb
    ser1 = ser[ser >= MIN_SCORE]
    self.assertTrue(all(ser.eq(ser1)))
    some_true = [feature_set_collection.FEATURE_SEPARATOR
        in f for f in ser.index]
    self.assertTrue(any(some_true))

  def test_ser_sbfset(self):
    if IGNORE_TEST:
      return
    ser = self.collection.ser_sbfset
    num_feature = len(ANALYZER.features)
    expected = num_feature*(num_feature-1)/2 + num_feature
    self.assertEqual(expected, len(ser))
    #
    trues = [(v <= 1) and (v >= 0) for v in ser]
    self.assertTrue(all(trues))
    #
    fsets = [FeatureSet(i) for i in ser.index]
    features = set([])
    for fset in fsets:
      features = features.union(fset.set)
      self.assertGreater(len(
          self.collection._analyzer.df_X[features]), 0)

  def testSerialize(self):
    if IGNORE_TEST:
      return
    self.collection.serialize(TEST_SERIALIZE_DIR)
    for stg in feature_set_collection.COMPUTES:
      path = os.path.join(TEST_SERIALIZE_DIR,
          "%s.csv" % stg)
      self.assertTrue(os.path.isfile(path))
    path = os.path.join(TEST_SERIALIZE_DIR,
        feature_set_collection.MISC_PCL)
    self.assertTrue(os.path.isfile(path))

  def testDeserialize(self):
    if IGNORE_TEST:
      return
    self.collection.serialize(TEST_SERIALIZE_DIR)
    collection = FeatureSetCollection.deserialize(
        TEST_SERIALIZE_DIR)
    self.assertEqual(collection._min_score,
       self.collection._min_score)
    self.assertEqual(len(self.collection.ser_comb),
        len(collection.ser_comb))
    self.assertEqual(len(self.collection.ser_sbfset),
        len(collection.ser_sbfset))

  def testProfileCollection(self):
    if IGNORE_TEST:
      return
    fset_stg = self.collection.ser_comb.index.tolist()[0]
    fset = FeatureSet(fset_stg)
    df = self.collection.profileFset(fset)
    columns = [cn.SUM, cn.PREDICTED, cn.CLASS,
        cn.INTERCEPT]
    columns.extend(fset.list)
    self.assertTrue(helpers.isValidDataFrame(df,
        expected_columns=columns))

  def testplotProfileFset(self):
    if IGNORE_TEST:
      return
    # Smoke test
    fset_stg = self.collection.ser_comb.index.tolist()[0]
    fset = FeatureSet(fset_stg)
    self.collection.plotProfileFset(fset,
        is_plot=IS_PLOT, title="Class 1")

  def testplotProfileFsets(self):
    if IGNORE_TEST:
      return
    # Smoke test
    fset_stgs = self.collection.ser_comb.index.tolist()
    self.collection.plotProfileFsets(fset_stgs,
        is_plot=IS_PLOT)
 


if __name__ == '__main__':
  unittest.main()


