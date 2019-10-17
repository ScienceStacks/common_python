'''Wrapper for a Tellurium model.'''

import common_python.tellurium.constants as cn
import common_python.tellurium.util as util

import tellurium as te
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Globals
runner = None


############ CLASSES ######################
class Model(object):
  """
  Abstraction for creating a simulation that can be run repeatedly
  with making changes to parameters.
  Key methods:
    runSimulation() - calculates values for species and times
    calcResiduals() - calculates residuals w.r.t. observations
  """

  def __init__(self, model_str, constants,
      simulation_time, num_points, parameters=None):
    """
    :param str model_str: Antimony model
    :param list-str constants: list of constants to fit in model
    :param int simulation_time: length of simulation
    :param int num_points: number of data points
    :param lmfit.Parameters parameters:
    """
    self.model_str = model_str
    self.road_runner = te.loada(self.model_str)
    self.constants = constants
    self.simulation_time = simulation_time
    self.num_points = num_points
    self.parameters = parameters
    self.df_data, self.ser_time = self.runSimulation(
        parameters=self.parameters)
    self.species = self.df_data.columns.tolist()

  def runSimulation(self, parameters=None):
    """
    Runs a simulation.
    :param Parameters parameters: If None, use existing values.
    :return pd.Series, pd.DataFrame: time, concentrations
    Notes:
      1. Updates self.df_data, self.ser_time
    """
    self.road_runner.reset()
    if parameters is not None:
      # Set the value of constants in the simulation
      param_dict = parameters.valuesdict()
      for constant in param_dict.keys():
        stmt = "self.road_runner.%s = param_dict['%s']" % (
            constant, constant)
        exec(stmt)
    #
    data = self.road_runner.simulate(0,
        self.simulation_time, self.num_points)
    # Construct the data frames
    df_alldata = pd.DataFrame(data)
    columns = [c[1:-1] for c in data.colnames]  # Eliminate square brackets
    columns[0] = cn.TIME
    df_alldata.columns = columns
    self.ser_time = df_alldata[cn.TIME]
    self.df_data = df_alldata[df_alldata.columns[1:]]
    return self.df_data, self.ser_time

  def calcResiduals(self, df_observation, model=None, parameters=None):
    """
    Calculates residuals for a model specified by the parameters.
    :param pd.DataFrame df_observation: observational data over time
    :param Model model: model to run. Default is parent.
    :param lmfit.Parameters parameters: parameters
    :return list-float: residuals
    """
    if model is None:
      model = self
    self.parameters = parameters
    df_species_data, _ = model.runSimulation(
        parameters=self.parameters)
    df_residuals = df_observation - df_species_data
    ser = util.dfToSer(df_residuals)
    return np.array(ser.tolist())

  def plotResiduals(self, df_observation, is_plot=True, **kwargs):
    residuals = self.calcResiduals(df_observation, **kwargs)
    plt.scatter(range(len(residuals)), residuals)
    plt.plot([0, len(residuals)], [0, 0],
        color='blue')
    plt.xlabel("Instance")
    plt.ylabel("Residual")
    if is_plot:
      plt.show()

  def plotData(self, df_observation=None, is_scatter=False,
      is_plot=True):
    """
    Plots the last run of the simulation (self.ser_time, self.df_data)
    """
    if df_observation is None:
      df_observation = self.df_data
    if is_scatter:
      for column in df_observation.columns:
        plt.scatter(self.ser_time, df_observation[column])
    else:
      plt.plot(self.ser_time, df_observation)
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.legend(self.species)

  def print(self):
    """
    Prints the model
    """
    print(self.model_str)

  # TODO: Implement a pickle serializaiton?
  def serialize(self):
    pass
    
