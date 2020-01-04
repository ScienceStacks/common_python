from common_python.experiment.experiment_harness  \
    import ExperimentHarness

import numpy as np
import pandas as pd
import sys
import unittest


IGNORE_TEST = True
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

def func(parm1=None, parm2=None):
  """
  Dummy calculation function.
  """
  return pd.DataFrame({
      COL_A: np.repeat(str(parm1) + str(parm2), SIZE),
      COL_B: np.repeat(str(parm2) + str(parm1), SIZE),
      })

class TestExperimentHarness(unittest.TestCase):

  def setUp(self):
    self.harness = ExperimentHarness(PARAM_DCT, func,
        out_path=OUT_PATH)
 
  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(self.harness._out_path, OUT_PATH)
    self.assertEqual(len(self.harness.df_result), 0)

  def testMakeRestoreDF(self):
    # TESTING
    df, completeds = self.harness._makeRestoreDF()
    self.assertEqual(len(df), 0)
    self.assertEqual(len(completeds), 0)
    #
    df = pd.DataFrame({
        PARAM1: 
    import pdb; pdb.set_trace()


if __name__ == '__main__':
  unittest.main()
