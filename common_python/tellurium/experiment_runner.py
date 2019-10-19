"Runs a tellurium experiment."

import common_python.tellurium.constants as cn
import common_python.tellurium.util as util
from common_python.tellurium.model import Model

import collections
import lmfit   # Fitting lib
import pandas as pd
import numpy as np
import random 

PARAMETER_DEFAULT_MIN = 0
PARAMETER_DEFAULT_MAX = 10
PARAMETER_DEFAULT_VALUE = 1

FitResult = collections.namedtuple('FitResult',
    'params rsq')


############ CLASSES ######################
class ExperimentRunner(Model):
  """
  Creates data for a Tellurium experiment by adding randomness
  to a simulation.
  Key methods:
    fit() - fit parameters to experimental data
  """

  def __init__(self, model_str, constants, simulation_time, num_points, 
      noise_std=0.5, indices=None,
      **kwargs):
    """
    :param str model_str: Antimony model
    :param list-str constants: list of constants to fit in model
    :param int simulation_time: length of simulation
    :param int num_points: number of data points
    :param float noise_std: std used to generate observations
                            if df_observation is None
    :param list-int indices: values used in parameter estimation
    :param dict kwargs: keyword arguments for Model
    """
    super().__init__(model_str, constants, simulation_time, 
        num_points, **kwargs)
    self.noise_std = noise_std
    self.df_observation, self.ser_time = self.makeObservations()
    if indices is None:
      indices = [i for i in range(self.num_points)]
    self.indices = indices

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
    :return FitResult: parameters (DF), rsq(float)
        DF: columns species; rows are cn.MEAN, cn.STD
    GLOBALS
      self.parameters - set to final fitted value
    """
    def calcResiduals(parameters):
      residuals = self.calcResiduals(
          self.df_observation, parameters=parameters,
          indices=self.indices)[0]
      return residuals
    #
    estimates = {}
    for constant in self.constants:
        estimates[constant] = []  # Initialize to empty list
    # Do the analysis multiple times with different observations
    for _ in range(count):
      self.df_observation, self_df_time = self.makeObservations()
      if parameters is None:
        parameters = self._makeParameters()
      # Create the minimizer
      fitter = lmfit.Minimizer(calcResiduals, parameters)
      result = fitter.minimize (method=method)
      for constant in self.constants:
        estimates[constant].append(
           result.params.get(constant).value)
    df_estimates = pd.DataFrame(estimates)
    df_result = pd.DataFrame()
    df_result[cn.MEAN] = df_estimates.mean(axis=0)
    df_result[cn.STD] = df_estimates.std(axis=0)
    self.parameters = self._makeParameters(
        df_result[cn.MEAN].tolist())
    rsq = self.calcResiduals(
          self.df_observation, parameters=parameters,
          indices=self.indices)[1]
    fit_result = FitResult(params=df_result, rsq=rsq)
    return fit_result

  def _makeParameters(self, values=None, mins=None, maxs=None):
    """
    :return Parameters: parameters for constants
    """
    def get(a_list, idx, default):
      if a_list is None:
        return default
      else:
        return a_list[idx]
    #
    parameters = lmfit.Parameters()
    for idx, constant in enumerate(self.constants):
        parameters.add(constant,
            value=get(values, idx, PARAMETER_DEFAULT_VALUE),
            min=get(mins, idx, PARAMETER_DEFAULT_MIN),
            max=get(maxs, idx, PARAMETER_DEFAULT_MAX))
    return parameters
