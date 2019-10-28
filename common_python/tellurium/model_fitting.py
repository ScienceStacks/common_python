'''Cross validation codes.'''

"""
kwargs: keyword arguments to runSimulation
order of positional arguments: obs_data, model, parameters
"""

import lmfit   # Fitting lib
import matplotlib.pyplot as plt
import numpy as np
import math
import random 
import tellurium as te



############## CONSTANTS ######################
# Default simulation model
MODEL = """
     A -> B; k1*A
     B -> C; k2*B
      
     A = 5;
     B = 0;
     C = 0;
     k1 = 0.1
     k2 = 0.2
"""
CONSTANTS = ['k1', 'k2']
NOISE_STD = 0.5
NUM_POINTS = 10
KWARGS_NUM_POINTS = "num_points"
PARAMETERS = lmfit.Parameters()
PARAMETERS.add('k1', value=1, min=0, max=10)
PARAMETERS.add('k2', value=1, min=0, max=10)
ROAD_RUNNER = None
SIM_TIME = 30

# Default parameters values
DF_CONFIDENCE_INTERVAL = (5, 95)
DF_BOOTSTRAP_COUNT = 5
DF_NUM_FOLDS = 3
DF_METHOD = "least_squares"


############## FUNCTIONS ######################
def reshapeData(matrix, indices=None):
  """
  Re-structures matrix as an array for just the rows
  in indices.
  :param array matrix: matrix of data
  :param list-int indices: indexes to include in reshape
  """
  if indices is None:
    nrows = np.shape(matrix)[0]
    indices = range(nrows)
  num_columns = np.shape(matrix)[1]
  trimmed_matrix = matrix[indices, :]
  return np.reshape(trimmed_matrix, num_columns*len(indices))

def arrayDifference(matrix1, matrix2, indices=None):
  """
  Calculates matrix1 - matrix2 as a nX1 array for the rows
  specified in indices.
  """
  array1 = reshapeData(matrix1, indices=indices)
  array2 = reshapeData(matrix2, indices=indices)
  return (array1 - array2)

def calcRsq(observations, estimates, indices=None):
  """
  Computes RSQ for simulation results.
  :param 2d-np.array observations: non-time values
  :param 2d-np.array estimates: non-time values
  :param list-int indices:
  :return float:
  """
  array_residuals = arrayDifference(observations, estimates,
      indices=indices)
  array_observations = reshapeData(observations, indices=indices)
  return 1 - np.var(array_residuals)/np.var(array_observations)

def makeParameters(constants=CONSTANTS, values=1, mins=0, maxs=10):
  """
  Constructs parameters for the constants provided.
  :param list-str constants: names of parameters
  :param list-float values: initial value of parameter
                            if not list, the value for list
  :param list-float mins: minimum values
                            if not list, the value for list
  :param list-float maxs: maximum values
                            if not list, the value for list
  """
  def makeList(val, list_length):
    if isinstance(val, list):
      return val
    else:
      return list(np.repeat(val, list_length))
  # 
  values = makeList(values, len(constants))
  mins = makeList(mins, len(constants))
  maxs = makeList(maxs, len(constants))
  parameters = lmfit.Parameters()
  for idx, constant in enumerate(constants):
    parameters.add(constant,
        value=values[idx], min=mins[idx], max=maxs[idx])
  return parameters

def makeAverageParameters(list_parameters):
  """
  Averages the values of parameters in a list.
  :param list-lmfit.Parameters list_parameters:
  :return lmfit.Parameters:
  """
  result_parameters = lmfit.Parameters()
  names = list_parameters[0].valuesdict().keys()
  for name in names:
    values = []
    for parameters in list_parameters:
      values.append(parameters.valuesdict()[name])
    result_parameters.add(name, value=np.mean(values))
  return result_parameters

def runSimulation(sim_time=SIM_TIME, 
    num_points=NUM_POINTS, parameters=None,
    road_runner=ROAD_RUNNER, model=MODEL):
  """
  Runs the simulation model rr for the parameters.
  :param int sim_time: time to run the simulation
  :param int num_points: number of timepoints simulated
  :param lmfit.Parameters parameters:
  :param ExtendedRoadRunner road_runner:
  :param str model:
  :return named_array:
  """
  if road_runner is None:
     road_runner = te.loada(model)
  else:
    road_runner.reset()
  if parameters is not None:
    parameter_dict = parameters.valuesdict()
    # Set the simulation constants for all parameters
    for constant in parameter_dict.keys():
      stmt = "road_runner.%s = float(parameter_dict['%s'])" % (
          constant, constant)
      try:
        exec(stmt)
      except:
        import pdb; pdb.set_trace()
  return road_runner.simulate (0, sim_time, num_points)

def plotTimeSeries(data, is_scatter=False, title="", 
    columns=None, is_plot=True):
  """
  Constructs a time series plot of simulation data.
  :param array data: first column is time
  :param bool is_scatter: do a scatter plot
  :param str title: plot title
  """
  if is_scatter:
    plt.plot (data[:, 0], data[:, 1:], marker='*', linestyle='None')
  else:
    plt.plot (data[:, 0], data[:, 1:])
  plt.title(title)
  plt.xlabel("Time")
  plt.ylabel("Concentration")
  if columns is None:
    columns = np.repeat("", np.shape(data)[1])
  plt.legend(columns)
  if is_plot:
    plt.show() 

def foldGenerator(num_points, num_folds):
  """
  Creates generator for test and training data indices.
  :param int num_points: number of data points
  :param int num_folds: number of folds
  :return iterable: Each iteration produces a tuple
                    First element: training indices
                    Second element: test indices
  """
  indices = range(num_points)
  for remainder in range(num_folds):
    test_indices = []
    for idx in indices:
      if idx % num_folds == remainder:
        test_indices.append(idx)
    train_indices = np.array(
        list(set(indices).difference(test_indices)))
    test_indices = np.array(test_indices)
    yield train_indices, test_indices

def makeObservations(sim_time=SIM_TIME, num_points=NUM_POINTS,
    noise_std=NOISE_STD, **kwargs):
  """
  Creates synthetic observations.
  :param int sim_time: time to run the simulation
  :param int num_points: number of timepoints simulated
  :param float noise_std: Standard deviation for random noise
  :param dict kwargs: keyword parameters used by runSimulation
  :return namedarray: simulation results with randomness
  """
  # Create true values
  data = runSimulation(sim_time=sim_time, num_points=num_points,
      **kwargs)
  num_cols = len(data.colnames)
  # Add randomness
  for i in range (num_points):
    for j in range(1, num_cols):
      data[i, j] = max(data[i, j]  \
          + np.random.normal(0, noise_std, 1), 0)
  return data

def calcSimulationResiduals(obs_data, parameters, indices=None, **kwargs):
  """
  Runs a simulation with the specified parameters and calculates residuals
  for the train_indices.
  :param array obs_data: matrix of data, first col is time.
  :param lmfit.Parameters parameters:
  :param list-int indices: indices for which calculation is done
                           if None, then all.
  :param dict kwargs: optional parameters passed to simulation
  """
  if not KWARGS_NUM_POINTS in kwargs.keys():
    kwargs[KWARGS_NUM_POINTS] = np.shape(obs_data)[0]
  sim_data = runSimulation(parameters=parameters,
      **kwargs)
  length = np.shape(sim_data)[0]
  residuals = arrayDifference(obs_data[:, 1:], sim_data[:, 1:],
      indices=indices)
  return residuals

def fit(obs_data, indices=None, parameters=PARAMETERS, method='leastsq',
    **kwargs):
  """
  Does a fit of the model to the observations.
  :param ndarray obs_data: matrix of observed values with time
                           as the first column
  :param list-int indices: indices on which fit is performed
  :param lmfit.Parameters parameters: parameters fit
  :param str method: optimization method
  :param dict kwargs: optional parameters passed to runSimulation
  :return lmfit.Parameters:
  """
  def calcLmfitResiduals(parameters):
    return calcSimulationResiduals(obs_data, parameters,
        indices, **kwargs)
  #
  # Estimate the parameters for this fold
  fitter = lmfit.Minimizer(calcLmfitResiduals, parameters)
  fitter_result = fitter.minimize(method=method)
  return fitter_result.params

def crossValidate(obs_data, method=DF_METHOD,
    sim_time=SIM_TIME,
    num_points=None, parameters=PARAMETERS,
    num_folds=3, **kwargs):
  """
  Performs cross validation on an antimony model.
  :param ndarray obs_data: data to fit; 
                           columns are species; 
                           rows are time instances
                           first column is time
  :param int sim_time: length of simulation run
  :param int num_points: number of time points produced.
  :param lmfit.Parameters: parameters to be estimated
  :param dict kwargs: optional arguments used in simulation
  :return list-lmfit.Parameters, list-float: parameters and RSQ from folds
  """
  if num_points is None:
    num_points = np.shape(obs_data)[0]
  # Iterate for for folds
  fold_generator = foldGenerator(num_points, num_folds)
  result_parameters = []
  result_rsqs = []
  for train_indices, test_indices in fold_generator:
    # This function is defined inside the loop because it references a loop variable
    new_parameters = parameters.copy()
    fitted_parameters = fit(obs_data, method=method,
      indices=train_indices, parameters=new_parameters,
      sim_time=SIM_TIME, num_points=num_points,
      **kwargs)
    result_parameters.append(fitted_parameters)
    # Run the simulation using
    # the parameters estimated using the training data.
    test_estimates = runSimulation(sim_time=sim_time,
        num_points=num_points,
        parameters=fitted_parameters, **kwargs)
    # Calculate RSQ
    rsq = calcRsq(obs_data[:, 1:], test_estimates[:, 1:],
        indices=test_indices)
    result_rsqs.append(rsq)
  return result_parameters, result_rsqs

def makeResidualsMatrix(obs_data, model, parameters, **kwargs):
  """
  Calculate residuals for each chemical species.
  :param np.array obs_data: matrix of observations; first column is time; number of rows is num_points
  :param lmfit.Parameters parameters:
  :param dict kwargs: optional arguments to runSimulation
  :return np.array: matrix of residuals; columns are chemical species
  """
  data = runSimulation(parameters=parameters, model=model, **kwargs)
  residuals = calcSimulationResiduals(obs_data, parameters,
      model=model, **kwargs)
  # Reshape the residuals by species
  rr = te.loada(model)
  num_species = rr.getNumFloatingSpecies()
  nrows = int(len(residuals) / num_species)
  return np.reshape(residuals, (nrows, num_species))

def makeSyntheticObservations(residual_matrix, **kwargs):
  """
  Constructs synthetic observations for the model.
  :param np.array residual_matrix: matrix of residuals; columns are species; number of rows is num_points
  :param dict kwargs: optional arguments to runSimulation
  :return np.array: matrix; first column is time
  """
  model_data = runSimulation(**kwargs)
  data = model_data.copy()
  nrows, ncols = np.shape(data)
  for icol in range(1, ncols):  # Avoid the time column
    indices = np.random.randint(0, nrows, nrows)
    for irow in range(nrows):
      data[irow, icol] = max(data[irow, icol] + residual_matrix[indices[irow], icol-1], 0)
  return data

def doBootstrapWithResiduals(residuals_matrix, 
    method=DF_METHOD, count=DF_BOOTSTRAP_COUNT, **kwargs):
  """
  Performs bootstrapping by repeatedly generating synthetic observations.
  :param np.array residuals_matrix: no time col
  :param int count: number of iterations in bootstrap
  :param dict kwargs: optional arguments to runSimulation
  """
  list_parameters = []
  for _ in range(count):
      obs_data = makeSyntheticObservations(residuals_matrix, **kwargs)
      parameters = fit(obs_data, method=method, **kwargs)
      list_parameters.append(parameters)
  return list_parameters

def doBootstrap(obs_data, model, parameters, 
    method=DF_METHOD, count=DF_BOOTSTRAP_COUNT,
    confidence_limits=DF_CONFIDENCE_INTERVAL, **kwargs):
  """
  Performs bootstrapping by repeatedly generating synthetic observations,
  calculating the residuals as well.
  :param array obs_data: matrix of data, first col is time.
  :param str model:
  :param lmfit.Parameters parameters:
  :param int count: number of iterations in bootstrap
  :param dict kwargs: optional arguments to runSimulation
  :return dict: confidence limits
  """
  residual_matrix = makeResidualsMatrix(obs_data, model,
      parameters, **kwargs)
  list_parameters = doBootstrapWithResiduals(residual_matrix,
      method=method,
      count=count, model=model, parameters=parameters, **kwargs)
  return makeParameterStatistics(list_parameters,
      confidence_limits=confidence_limits)

def makeParameterStatistics(list_parameters,
    confidence_limits=DF_CONFIDENCE_INTERVAL):
  """
  Computes the mean and standard deviation of the parameters in a list of parameters.
  :param list-lmfit.Parameters
  :param (float, float) confidence_limits: if none, report mean and variance
  :return dict: key is the parameter name; value is the tuple (mean, stddev) or confidence limits
  """
  parameter_statistics = {}  # This is a dictionary that will have the parameter name as key, and mean, std as values
  parameter_names = list(list_parameters[0].valuesdict().keys())
  for name in parameter_names:
    parameter_statistics[name] = []  # We will accumulate values in this list
    for parameters in list_parameters:
      parameter_statistics[name].append(parameters.valuesdict()[name])
  # Calculate the statistics
  for name in parameter_statistics.keys():
    if confidence_limits is not None:
      parameter_statistics[name] = np.percentile(parameter_statistics[name], confidence_limits)
    else:
      mean = np.mean(parameter_statistics[name])
      std = np.std(parameter_statistics[name])
      std = std/np.sqrt(len(list_parameters))  # adjustments for the standard deviation of the mean
      parameter_statistics[name] = (mean, std)
  return parameter_statistics
