'''Wrapper for a Tellurium model.'''

import common_python.tellurium.constants as cn
import common_python.tellurium.util as util

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tellurium as te


############ CLASSES ######################
class Model(object):
  """
  Abstraction for creating a simulation that can be run repeatedly
  with making changes to parameters. Maintains the value of
  the data, time, and parameters
  Key methods:
    runSimulation() - calculates values for species and times
    calcResiduals() - calculates residuals w.r.t. observations
  Key state:
    self.parameters - parameter value changes
    self.ser_time - time
    self.df_data - species concentrations
    self.species - chemical species
  """

  def __init__(self, model_str, constants,
      simulation_time, num_points, parameters=lmfit.Parameters()):
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
    self.df_data, self.ser_time = self.runSimulation()
    self.species = self.df_data.columns.tolist()

  def runSimulation(self, parameters=None):
    """
    Runs a simulation.
    :param Parameters parameters: If None, use existing values.
    :return pd.Series, pd.DataFrame: time, concentrations
    Instance variables modified:
        self.df_data, self.ser_time, self.parameters
    """
    self.road_runner.reset()
    if parameters is not None:
      self.parameters = parameters
    # Set the value of constants in the simulation
    param_dict = self.parameters.valuesdict()
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

  def calcResiduals(self, df_observation, model=None, 
      parameters=None, indices=None):
    """
    Calculates residuals for a model specified by the parameters.
    :param pd.DataFrame df_observation: observational data over time
    :param Model model: model to run. Default is parent.
    :param lmfit.Parameters parameters: parameters
    :return list-float, float: residuals, rsq
    """
    def trimRows(df, indices):
      dfindices = [df.index.tolist()[i] for i in indices]
      return df.loc[dfindices, :]
    #
    if model is None:
      model = self
    if indices is None:
      indices = [i for i in range(self.num_points)]
    df_species_data, _ = model.runSimulation(parameters=parameters)
    df_species_data = trimRows(df_species_data, indices)
    df_observation = trimRows(df_observation, indices)
    df_residuals = df_observation - df_species_data
    ser_res = util.dfToSer(df_residuals)
    res = np.array(ser_res.tolist())
    ser_obs = util.dfToSer(df_observation)
    obs = np.array(ser_obs.tolist())
    rsq = 1 - np.var(res)/np.var(obs)
    return res, rsq

  def plotResiduals(self, df_observation, is_plot=True, **kwargs):
    residuals = self.calcResiduals(df_observation, **kwargs)[0]
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
    
