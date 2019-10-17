from common_python.tellurium import experiment_runner

import lmfit
import numpy as np
import pandas as pd
import unittest


MODEL = """
     A -> B; k1*A
      
     A = 50; 
     B = 0;
     C = 0;
     k1 = 0.15
"""
CONSTANTS = ['k1']
COLUMNS = ['time', 'A', 'B']
#
MODEL = """
     A -> B; k1*A
     B -> C; k2*B
      
     A = 50; 
     B = 0;
     C = 0;
     k1 = 0.15
     k2 = 0.25
"""
CONSTANTS = ['k1', 'k2']
COLUMNS = ['time', 'A', 'B', 'C']
#
MODEL1 = """
     A -> B; k1*A
      
     A = 50; 
     B = 0;
     k1 = 0.15
"""
CONSTANT1S = ['k1']
SIMULATION_TIME = 30
NUM_POINTS = 5
COLUMN1S = ['time', 'A', 'B']


class TestExperimentRunner(unittest.TestCase):

  def testConstructor(self):
    runner = experiment_runner.ExperimentRunner(MODEL, CONSTANTS,
        SIMULATION_TIME, NUM_POINTS)
    trues = [c in COLUMNS for c in runner.df_observation.columns]
    assert(all(trues))
    assert(len(runner.df_observation) > 0)
  
  def testGenerateObservations(self):
    runner = experiment_runner.ExperimentRunner(MODEL, CONSTANTS,
        SIMULATION_TIME, NUM_POINTS)
    df, _ = runner.makeObservations()
    assert(len(set(df.columns).symmetric_difference(
        runner.df_observation.columns)) == 0)
  
  def testFit(self):
    for constants, model in [(CONSTANTS, MODEL), (CONSTANT1S, MODEL1)]:
      runner = experiment_runner.ExperimentRunner(model, constants,
          SIMULATION_TIME, NUM_POINTS, noise_std=0.0)
      df = runner.fit(count=20)
      assert(len(df.columns) == 2)
      assert(len(df) == len(constants))


if __name__ == '__main__':
  unittest.main()
