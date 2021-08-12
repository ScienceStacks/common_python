from common_python.tests.classifier import helpers as test_helpers
from common_python.classifier import feature_set
from common_python.classifier.feature_set_collection  \
    import FeatureSetCollection
from common_python.classifier.feature_set  \
    import FeatureSet
from common_python.classifier  \
    import feature_set_collection
from common_python.testing import helpers
from common_python import constants as cn
from common_python.util.persister import Persister
from common_python.classifier import feature_analyzer
from common import constants as xcn

import copy
import matplotlib.pyplot as plt
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
TEST_PERSISTER_PATH = os.path.join(TEST_DIR,
    "TEST_FEATURE_SET_COLLECTION.pcl")
PERSISTER = Persister(TEST_PERSISTER_PATH)
ANALYZER = test_helpers.getFeatureAnalyzer()
MIN_SCORE = 0.9
DF_X = ANALYZER.df_X
SER_Y = ANALYZER.ser_y
sorted_index = sorted(DF_X.index.tolist(),
     key=feature_set.SORT_FUNC)
DF_X = DF_X.loc[sorted_index, :]
SER_Y = SER_Y.loc[sorted_index]
COLLECTION = FeatureSetCollection.deserialize(
        TEST_SERIALIZE_DIR)


##########################################
class TestFeatureSetCollection(unittest.TestCase):

  def setUp(self):
    self.df_X = copy.deepcopy(DF_X)
    self.ser_y = copy.deepcopy(SER_Y)
    self.collection = copy.deepcopy(COLLECTION)

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
    if not all(ser.eq(ser1)):
      import pdb; pdb.set_trace() # Catch intermitent bug
    self.assertTrue(all(ser.eq(ser1)))
    some_true = [cn.FEATURE_SEPARATOR
        in f for f in ser.index]
    self.assertTrue(any(some_true))

  def test_mkSerComb(self):
    if IGNORE_TEST:
      return
    self.collection._ser_comb = None
    self.collection._min_score = 0.6
    max_size = 2
    ser = self.collection._mkSerComb(max_size=max_size, max_search=10)
    self.assertEqual(len(ser), max_size)
    trues = [v >= 0.7 for v in ser.to_numpy()]
    self.assertTrue(all(trues))

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
      if stg != feature_set_collection.DF_FV:
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
    self.assertTrue(collection.df_fv is not None)
    self.assertEqual(collection._min_score,
       self.collection._min_score)
    self.assertEqual(len(self.collection.ser_comb),
        len(collection.ser_comb))
    self.assertEqual(len(self.collection.ser_sbfset),
        len(collection.ser_sbfset))

  def testPlotProfileInstance(self):
    if IGNORE_TEST:
      return
    # Smoke test
    fset_stgs = self.collection.ser_comb.index.tolist()
    self.collection.plotProfileInstance(fset_stgs,
        is_plot=IS_PLOT)

  def testGetFVEvaluations(self):
    if IGNORE_TEST:
      return
    def test(instance, sl_range):
      ser_x = self.df_X.loc[instance]
      df = self.collection.getFVEvaluations(
          ser_x, max_sl=1)
      fvs = df[cn.FEATURE_VECTOR]
      sls = df[cn.SIGLVL]
      sl = np.abs(sls[0])
      self.assertGreater(sl, sl_range[0])
      self.assertLess(sl, sl_range[1])
    #
    test("T1.1", [0.2, 1.0])
    test("T8.1", [0, 10e-3])

  def testPlotEvaluate(self):
    if IGNORE_TEST:
      return
    for instance in ["T1.1", "T2.1"]:
      ser_x = self.df_X.loc[instance]
      self.collection.plotEvaluate(ser_x, max_sl=1,
          label_xoffset=-0.1,
          title=instance, is_plot=IS_PLOT)

  def testPlotFullProile(self):
    if IGNORE_TEST:
      return
    STATES = range(6)
    DATA_PATH = xcn.PROJECT_DIR
    for directory in ["data", "feature_analyzer"]:
        DATA_PATH = os.path.join(DATA_PATH, directory)
    DATA_PATH_PAT = os.path.join(DATA_PATH, "%d") 
    ANALYZER_DCT = feature_analyzer.deserialize(
       {s: DATA_PATH_PAT % s for s in STATES})
    ANALYZERS = ANALYZER_DCT.values()
    COLLECTION_DCT = {s: feature_set_collection.
        FeatureSetCollection.deserialize(
        DATA_PATH_PAT % s) for s in STATES}
    def fullProfile(ser_X, title=""):
        num_row = 2
        num_col = 3
        fig, axes = plt.subplots(num_row, num_col,
            figsize=(16, 10))
        for idx, state in enumerate(STATES):
          row = int(idx/num_col)
          col = idx % num_col
          collection = COLLECTION_DCT[state]
          if row == 0:
              label_xoffset = -0.1
          else:
              label_xoffset = 0.1
          collection.plotEvaluate(ser_X, 
              ax=axes[row, col], is_plot=False,
              title = "State %d" % idx,
              label_xoffset=label_xoffset)
        fig.suptitle(title, fontsize=16)
        if IS_PLOT:
          plt.show()
    #
    instance = "T3.0"
    ser_X = DF_X.loc[instance]
    fullProfile(ser_X, title=instance)

  def testMakeCase(self):
    if IGNORE_TEST:
      return
    df = self.collection._makeFeatureVector()
    self.assertTrue(helpers.isValidDataFrame(df,
        expected_columns=[cn.FEATURE_SET,
        cn.NUM_ZERO, cn.FEATURE_VECTOR]))
    #
    self.assertTrue(df.equals(self.collection.df_fv))

  def testGetNumZero(self):
    if IGNORE_TEST:
      return
    instance = "T3.0"
    ser_X = DF_X.loc[instance]
    fsets, num_zeroes = self.collection._getNumZero(ser_X)
    self.assertEqual(len(fsets), len(num_zeroes))
    #
    new_fsets, new_num_zeroes =  \
         self.collection._getNumZero(ser_X, max_sl=0.001)
    self.assertEqual(len(new_fsets), len(new_num_zeroes))
    self.assertGreater(len(fsets), len(new_fsets))

  def testPlotEvaluateHistogram(self):
    if IGNORE_TEST:
      return
    for instance in ["T1.1", "T3.0"]:
      ser_X = DF_X.loc[instance]
      self.collection.plotEvaluateHistogram(ser_X,
          is_plot=IS_PLOT, max_sl=0.01)



if __name__ == '__main__':
  unittest.main()


