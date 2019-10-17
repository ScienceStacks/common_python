"Runs a tellurium model"

import common_python.tellurium.constants as cn
import common_python.tellurium.util as util
from common_python.tellurium.model_maker import ModelMaker

import lmfit   # Fitting lib
import random 

# Globals
runner = None

############ FUNCTIONS ######################
def residuals(parameters):
  """
  Computes residuals from the parameters using the global runner
  Residuals are computed based on the values of chemical species
  in the model.
  :param lmfit.Parameter parameters:
  """
  df_sim_data, _ = runner.runSimulation(parameters=parameters)
  df_residuals = runner.df_observation - df_sim_data
  ser = util.dfToSer(df_residuals)
  return np.array(ser.tolist())


############ CLASSES ######################
class ModelRunner(Model):

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
    self.df_observation, self.ser_time = self.generateObservations()
    self.df_noisey = None

  def generateObservations(self, parameters=None, std=None):
    """
    Creates random observations by adding normally distributed
    noise.
    :param float std: if none, use constructor
    :return pd.DataFrame, ser_times: noisey data, time
    """
    df_data, ser_time = self.runSimulation(parameters=parameters)
    if std is None:
      std = self.noise_std
    df_rand = pd.DataFrame(np.random.normal(0, std,
         (len(df_data),len(df_data.columns))))
    df_rand.columns = df_data.columns
    df = df_data + df_rand
    df = df.applymap(lambda v: 0 if v < 0 else v)
    return df, ser_time
  
  # FIXME: Allow for specifying min/max for constants 
  def fit(self, count=1, method='leastsq', std=None, func=None):
    """
    Performs multiple fits.
    :param int count: Number of fits to do, each with different
                      noisey data
    :return pd.DataFrame: columns species; rows are cn.MEAN, cn.STD
    Assigns value to df_noisey to communicate with func
    """
    if func is None:
      global runner
      runner = self
      func = residuals
    if std is None:
      std = self.noise_std
    #
    estimates = {}
    for constant in self.constants:
        estimates[constant] = []  # Initialize to empty list
    # Do the analysis multiple times with different observations
    for _ in range(count):
      self.df_observation, self_df_time = self.generateObservations()
      parameters = lmfit.Parameters()
      for constant in self.constants:
          parameters.add(constant, value=1, min=0, max=10)
      # Create the minimizer
      fitter = lmfit.Minimizer(func, parameters)
      result = fitter.minimize (method=method)
      for constant in self.constants:
        estimates[constant].append(
           result.params.get(constant).value)
    df_estimates = pd.DataFrame(estimates)
    df_result = pd.DataFrame()
    df_result[cn.MEAN] = df_estimates.mean(axis=0)
    df_result[cn.STD] = df_estimates.std(axis=0)
    return df_result
        
