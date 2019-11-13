'''Tests for modeling game.'''

from common_python.util import util
util.addPath("common_python", 
    sub_dirs=["common_python", "tellurium"])
import model_fitting as mf
import modeling_game as mg

import lmfit
import numpy as np
import pandas as pd

CSV_FILE = "common_python/tests/tellurium/wild.csv"


def testMakeParameters():
  def test(constants, values=None):
    parameters = mg.makeParameters(constants, values=values)
    assert(len(constants) == len(parameters.valuesdict().keys()))
    if values is not None:
      trues = [v == parameters.valuesdict()[c] for
          c, v in zip(constants, values)]
      assert(all(trues))
  #
  constants = ["Vm1", "d_protein2"]
  test(constants) 
  test(constants, values=[0.5, 0.01]) 

def testDoBootstrap3():
  constants = ["Vm"]
  sim_time = 300
  num_points = int(sim_time/10)
  parameters = mg.makeParameters(constants)
  df_rnaseq = pd.read_csv(CSV_FILE)
  df_obs = df_rnaseq.loc[df_rnaseq.index[range(num_points)], :]
  statistic_dict = mf.doBootstrap(df_obs, model, parameters,
                                             num_points=num_points, 
                                             method=mf.ME_BOTH,
                                             sim_time=sim_time)
   
  
if __name__ == '__main__':
  testDoBootstrap3()
  if True:
    testMakeParameters()
  print("OK.")
