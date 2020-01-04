from common_python.experiment.experiment_harness  \
    import ExperimentHarness
from common_python.testing import helpers

import os
import numpy as np
import pandas as pd
import sys
import unittest


IGNORE_TEST = False
COL_A = "a"
COL_B = "b"
SIZE = 3
PARAM1 = "param1"
PARAM2 = "param2"
PARAM_DCT = {
    PARAM1: ["a", "b", "c"],
    PARAM2: ["x", "y", "z"],
    }
OUT_PATH = "test_experiment_harness.csv"

def func(param1=None, param2=None):
  """
  Dummy calculation function.
  """
  return pd.DataFrame({
      COL_A: np.repeat(str(param1) + str(param2), SIZE),
      COL_B: np.repeat(str(param2) + str(param1), SIZE),
      })

class TestExperimentHarness(unittest.TestCase):

  def cleanState(self):
    if os.path.isfile(OUT_PATH):
      os.remove(OUT_PATH)

  def setUp(self):
    self.cleanState()
    self.harness = ExperimentHarness(PARAM_DCT, func,
        out_path=OUT_PATH, update_rpt=1)

  def tearDown(self):
    self.cleanState()
 
  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(self.harness._out_path, OUT_PATH)
    self.assertEqual(len(self.harness.df_result), 0)

  def testMakeRestoreDF(self):
    if IGNORE_TEST:
      return
    df, completeds = self.harness._makeRestoreDF()
    self.assertEqual(len(df), 0)
    self.assertEqual(len(completeds), 0)
    #
    VALUES = ["xx", "yy"]
    df_initial = func(param1=VALUES[0], param2=VALUES[1])
    df_initial[PARAM1] = np.repeat("xx", len(df_initial))
    df_initial[PARAM2] = np.repeat("yy", len(df_initial))
    df_initial.to_csv(OUT_PATH)
    df, completeds = self.harness._makeRestoreDF()
    self.assertTrue(completeds[0] == tuple(VALUES))
    self.assertTrue(df.equals(df_initial))

  def testRun(self):
    if IGNORE_TEST:
      return
    df = self.harness.run()
    values = [len(v) for v in PARAM_DCT.values()]
    expected_length = SIZE * np.prod(values)
    self.assertEqual(expected_length, len(df))
    expected_columns = [COL_A, COL_B, PARAM1, PARAM2]
    self.assertTrue(helpers.isValidDataFrame(df,
        expected_columns=expected_columns))


if __name__ == '__main__':
  unittest.main()
