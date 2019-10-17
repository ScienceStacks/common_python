"Runs a tellurium experiment."

import common_python.tellurium.constants as cn
import common_python.tellurium.util as util
from common_python.tellurium.model import Model

import lmfit   # Fitting lib
import pandas as pd
import numpy as np
import random 


############ CLASSES ######################
class ExperimentRunner(Model):
  """
  Creates data for a Tellurium experiment by adding randomness
  to a simulation.
  Key methods:
    fit() - fit parameters to experimental data
  """

  def __init__(self, model_str, constants,
      simulation_time, num_points, noise_std=0.5):
    """
    :param str model_str: Antimony model
    :param list-str constants: list of constants to fit in model
    :param int simulation_time: length of simulation
    :param int num_points: number of data points
    """
    super().__init__(model_str, constants, simulation_time, num_points)
    self.noise_std = noise_std
    self.df_observation, self.ser_time = self.makeObservations()
    self.df_noisey = None

  def makeObservations(self, parameters=None):
    """
    Creates random observations by adding normally distributed
    noise.
    :return pd.DataFrame, ser_times: noisey data, time
    """
    df_data, ser_time = self.runSimulation(parameters=parameters)
    df_rand = pd.DataFrame(np.random.normal(0, self.noise_std,
         (len(df_data),len(df_data.columns))))
    df_rand.columns = df_data.columns
    df = df_data + df_rand
    df = df.applymap(lambda v: 0 if v < 0 else v)
    return df, ser_time
  
  def fit(self, parameters=None, count=1, method='leastsq'):
    """
    Performs multiple fits.
    :param lmfit.Parameters parameters: parameters to fit or use default.
                                        default: set to 1
    :param int count: Number of fits to do, each with different
                      noisey data
    :return pd.DataFrame: columns species; rows are cn.MEAN, cn.STD
    """
    def residuals(parameters):
      return self.calcResiduals(self.df_observation, parameters=parameters)
    #
    estimates = {}
    for constant in self.constants:
        estimates[constant] = []  # Initialize to empty list
    # Do the analysis multiple times with different observations
    for _ in range(count):
      self.df_observation, self_df_time = self.makeObservations()
      if parameters is None:
        parameters = lmfit.Parameters()
        for constant in self.constants:
            parameters.add(constant, value=1, min=0, max=10)
      # Create the minimizer
      fitter = lmfit.Minimizer(residuals, parameters)
      result = fitter.minimize (method=method)
      for constant in self.constants:
        estimates[constant].append(
           result.params.get(constant).value)
    df_estimates = pd.DataFrame(estimates)
    df_result = pd.DataFrame()
    df_result[cn.MEAN] = df_estimates.mean(axis=0)
    df_result[cn.STD] = df_estimates.std(axis=0)
    return df_result
