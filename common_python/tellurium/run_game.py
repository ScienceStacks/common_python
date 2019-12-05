import lmfit
import numpy as np
import pandas as pd

import constants as cn
import gene_network as gn
import modeling_game as mg
import model_fitting as mf
import gene_analyzer as ga
import util


SIM_TIME = 1200
NUM_POINTS = 120
FULL_MODEL = "full_model.txt"
PARAM_FILE = "parameters.csv"
MRNA_FILE = "wild.csv"

def evaluate(model=FULL_MODEL, param_file=PARAM_FILE,
    mrna_file=MRNA_FILE):
  # Initializations
  with open(FULL_MODEL, "r") as fd:
    full_model = fd.readlines()
  full_model = "".join(full_model)
  #
  df_params = pd.read_csv(PARAM_FILE)
  parameters = mg.makeParameters(df_params[cn.NAME],
      df_params[cn.VALUE])
  # Run simulation
  df_mrna = pd.read_csv(MRNA_FILE)
  df_mrna = df_mrna.set_index(cn.TIME)
  df_mrna = df_mrna.drop(df_mrna.index[-1])
  fitted_parameters = mf.fit(df_mrna, model=full_model,
       parameters=parameters,
      sim_time=SIM_TIME, num_points=NUM_POINTS)
  # Simulate with fitted parameters and plot
  
  mg.plotSimulation(df_mrna, full_model, parameters=fitted_parameters,
      is_plot_observations=True, is_plot_model=True,
      is_plot_residuals=True, title=None)
