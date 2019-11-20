"""Analysis Codes for the Modeling Game for BIOE 498V."""

import sys
import os
import model_fitting as mf

import numpy as np
import tellurium as te
import matplotlib.pyplot as plt
import pandas as pd
import lmfit
import os
import re

# Symbols
# Column names
TIME = 'time'

TIME_TO_POINT = 10 # 1 point for every second
NUM_POINTS = 120
START_TIME = 0
END_TIME = 1200


################### Functions ####################
def makeDF(named_array, is_mrna=True, is_protein=True, is_input=False):
  """
  :param bool is_mrna: include mRNA in output
  :param bool is_protein: include proteins in output
  :param bool is_input: include input in output
  :return pd.DataFrame: Dataframe with time as index
  """
  def delColumn(df, string):
    for col in df.columns:
      if col[:len(string)] == string:
        del df[col]
  #
  df = pd.DataFrame(named_array)
  df.columns = named_array.colnames
  df = mf.cleanColumns(df)
  #
  if not is_mrna:
    delColumn(df, "mRNA")
  if not is_protein:
    delColumn(df, "P")
  if not is_input:
    delColumn(df, "INPUT")
  return df

def simulate(model, start_time=START_TIME, end_time=END_TIME, num_points=NUM_POINTS,
    **kwargs):
  """
  Runs a simulation for the modeling game. Allows for selection
  of the desired outputs.
  :param str model:
  :param dict kwargs: arguments for makeDF
  :return pd.DataFrame: Dataframe with time as index
  """
  rr = te.loada(model)
  data = rr.simulate(start_time, end_time, num_points)
  return makeDF(data)

def getRNASeq(csv_file):
  """
  Get RNA sequence data from a local csv file.
  :param str csv_file:
  :return pd.DataFrame: cleans column names; time is index.
  """
  path = os.path.join("data", csv_file)
  df = pd.read_csv(path)
  df[TIME] = [int(v) for v in df[TIME]]
  df = df.set_index(TIME)
  return df

def makeResiduals(model, df, **kwargs):
  """
  Calculates the residuals for the columns in common.
  :param str model:
  :param pd.DataFrame df: observational data compatible with
                          model simulation.
  :param dict kwargs: optional parameters for simulation
  """
  df_sim = simulate(model, **kwargs)
  columns = set(df_sim.columns).intersection(df.columns)
  return df[columns] - df_sim[columns]

def plotData(df_data, starttime=0, endtime=1200, title=""):
  last = int(endtime/TIME_TO_POINT)
  indices = df_data.index[0:last]
  plt.plot(indices, df_data.loc[indices,:])
  plt.xlabel("Time")
  plt.title(title)
  plt.legend(df_data.columns, loc="upper right")

def makeParameters(constants, values=None):
  """
  Creates parameters with the correct ranges based on their names.
  :param str constants: names of constant
  :param list-float values: initial values to use.
  """
  if isinstance(constants, str):
      constants = [constants]
  # mins and maxs for parameters by their initial string (up to a number)
  ranges_dict = {
      "Vm": (0.5, 2), 
      "K": (0.01, 0.03), 
      "L": (0.01, 0.03), 
      "H": (2, 8), 
      "a_protein": (0.05, 0.15), 
      "d_protein": (0.01, 0.03), 
      "d_mRNA": (0.5, 2)
      }
  parameters = lmfit.Parameters()
  for idx, constant in enumerate(constants):
    pfxs = re.findall(r"^\D+", constant)
    if len(pfxs) != 1:
        raise ValueError("Cannot find match for %s" % constant)
    pfx = pfxs[0]
    is_keyerror = False
    try:
      min_val, max_val = ranges_dict[pfx]
    except KeyError:
      is_keyerror = True
    if is_keyerror:
      raise ValueError("No value range defined for parameter type %s"
          % pfx)
    if values is None:
      initial_val = (min_val + max_val) / 2
    else:
      initial_val = values[idx]
    parameters.add(constant, value=initial_val, min=min_val, max=max_val)
  return parameters

def _selectRNA(df):
  if not isinstance(df, pd.DataFrame):
    df = makeDF(df, is_protein=False)
  columns = [ c for c in df.columns if "mRNA" in c]
  df_result = df.copy()
  return df_result[columns]

def plotSimulation(df_data, model, parameters=None, sim_time=1200):
  """
  Plots mRNA actual, predicted, and residuals.
  :param pd.DataFrame df_data: dataframe of observed data.
  :param str model: antimony model
  :param lmfit.Parameters parameters:
  :param int sim_time: length of time to simulate
  """
  num_points = int(sim_time/10)
  df_mrna = _selectRNA(df_data)
  sim_result = mf.runSimulation(model=model,sim_time=sim_time, 
      num_points=num_points, parameters=parameters)
  df_model = makeDF(sim_result.data, is_protein=False)
  df_res = df_mrna - df_model
  df_res = _selectRNA(df_res)
  for args in [(df_mrna, "Observations"), 
      (df_model, "Model"), (df_res, "Residuals")]:
    plt.figure()
    plotData(args[0], endtime=sim_time, title=args[1])

def runExperiment(df_data, model, parameters, sim_time=1200):
  """
  Runs an experiment in which the the Antimony model is fitted to
  mRNA data and plots are produced to evaluate the result.
  :param pd.DataFrame df_data: dataframe of observed data.
      time is index
  :param str model: antimony model
  :param lmfit.parameters/list-str parameters:
       Constants for which parameters are estimated
  :param int sim_time: length of the simulation 
  :return lmfit.Parameters:
  """
  num_points = int(sim_time/10)
  if not isinstance(parameters, lmfit.Parameters):
    parameters = makeParameters(parameters)
  df_mrna = _selectRNA(df_data)
  df_obs = df_mrna.loc[df_mrna.index[range(num_points)], :]
  df_obs = _selectRNA(df_obs)
  list_parameters, rsqs = mf.crossValidate(df_obs, model=model,
    parameters=parameters, num_points=num_points, method=mf.ME_BOTH,
    sim_time=sim_time, num_folds=3)
  parameters = mf.makeAverageParameters(list_parameters)
  plotSimulation(df_obs, model, parameters=parameters,
      sim_time=sim_time)
  return parameters
