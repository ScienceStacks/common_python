'''Cross validation codes.'''

"""
kwargs: keyword arguments to runSimulation
order of positional arguments: obs_data, model, parameters
"""

import lmfit   # Fitting lib
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import random 
import tellurium as te

TIME = "time"


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
def matrixToDF(matrix, columns=None):
  """
  Converts an array to a dataframe.
  :param np.arrayy matrix:
         pd.DataFrame    : already converted
  :return pd.DataFrame: Columns are variables w/o [, ].
                        index is time.
  """
  if isinstance(matrix, pd.DataFrame):
    return matrix
  df = pd.DataFrame(matrix)
  try:
    # Assign column names if present
    columns = [c[1:-1] for c in matrix.colnames]
    columns[0] = TIME
  except:
    pass
  if columns is not None:
    df.columns = columns
  if TIME in df.columns:
    df = df.set_index(TIME)
  return df

def matrixToDFWithoutTime(matrix, columns=None):
  """
  Converts an array to a dataframe, deleting time as a column.
  :param np.arrayy matrix:
         pd.DataFrame    : already converted
  :return pd.DataFrame:
  """
  if isinstance(matrix, pd.DataFrame):
    df = matrix.copy()
  else:
    df = matrixToDF(matrix, columns=columns)
  if TIME != df.index.name:
    df = df[df.columns.tolist()[1:]]
  return df

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
  :param matrix observations: non-time values
  :      pd.DataFrame       : no time column
  :param matrix estimates: non-time values
  :      pd.DataFrame       : no time column
  :param list-int indices:
  :return float:
  """
  df_obs = matrixToDF(observations)
  df_est = matrixToDF(estimates)
  if indices is None:
    indices = range(len(df_obs))
  #
  def makeSub(df):
    return df.loc[df.index[indices], :]
  #
  df_obs_sub = makeSub(df_obs)
  df_est_sub = makeSub(df_est)
  df_rsq = df_obs_sub - df_est_sub
  df_rsq = df_rsq*df_rsq
  arr_obs = np.reshape(df_obs_sub.values,
      len(indices)*len(df_obs.columns))
  rsq = 1 - df_rsq.sum().sum() / np.var(arr_obs)
  return rsq

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
    num_points=NUM_POINTS, road_runner=ROAD_RUNNER,
    **kwargs):
  """
  Runs the simulation model rr for the parameters.
  :param int sim_time: time to run the simulation
  :param int num_points: number of timepoints simulated
  :param ExtendedRoadRunner road_runner:
  :param dict kwargs: parameters used in makeSimulation
  :return named_array:
  """
  if road_runner is None:
     road_runner = makeSimulation(**kwargs)
  else:
    road_runner.reset()
  return road_runner.simulate (0, sim_time, num_points)

def makeSimulation(parameters=None, model=MODEL):
  """
  Creates an road runner instance for the simulation.
  :param lmfit.Parameters parameters:
  :param str model:
  :return ExtendedRoadRunner:
  """
  road_runner = te.loada(model)
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
  return road_runner

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

def calcSimulationResiduals(obs_data, parameters,
    indices=None, columns=None, **kwargs):
  """
  Runs a simulation with the specified parameters and calculates residuals
  for the train_indices.
  :param array obs_data: matrix of data, first col is time.
         pd.DataFrame  : index is time
  :param lmfit.Parameters parameters:
  :param list-str columns: columns to use in residual calculations
  :param list-int indices: indices for which calculation is done
                           if None, then all.
  :param dict kwargs: optional parameters passed to simulation
  :return array:
  """
  df_obs = matrixToDFWithoutTime(obs_data)
  if not KWARGS_NUM_POINTS in kwargs.keys():
    kwargs[KWARGS_NUM_POINTS] = len(df_obs)
  raw_data = runSimulation(parameters=parameters,
      **kwargs)
  df_sim = matrixToDFWithoutTime(raw_data)
  if columns is not None:
    df_sim = df_sim[columns]
  #
  array_sim = df_sim.values
  array_obs = df_obs.values
  residuals = arrayDifference(array_obs, array_sim,
      indices=indices)
  return residuals

# TODO: Fix handling of obs_data columns. May not be
#       a named array, as in bootstrap.
def fit(obs_data, indices=None, parameters=PARAMETERS, 
    method='leastsq', columns=None,
    **kwargs):
  """
  Does a fit of the model to the observations.
  :param ndarray obs_data: matrix of observed values with time
                           as the first column
         pd.DataFrame    : time is index
  :param list-int indices: indices on which fit is performed
  :param lmfit.Parameters parameters: parameters fit
  :param str method: optimization method
  :param list-str columns: columns to use in fit
  :param dict kwargs: optional parameters passed to runSimulation
  :return lmfit.Parameters:
  """
  def calcLmfitResiduals(parameters):
    return calcSimulationResiduals(obs_data, parameters,
        indices, columns=columns, **kwargs)
  #
  # Estimate the parameters for this fold
  fitter = lmfit.Minimizer(calcLmfitResiduals, parameters)
  fitter_result = fitter.minimize(method=method)
  return fitter_result.params

def crossValidate(obs_data, method=DF_METHOD,
    sim_time=SIM_TIME,
    columns=None,
    num_points=None, parameters=PARAMETERS,
    num_folds=3, **kwargs):
  """
  Performs cross validation on an antimony model.
  :param ndarray obs_data: data to fit; 
                           columns are species; 
                           rows are time instances
                           first column is time
        pd.DataFrame    : time is index
  :param int sim_time: length of simulation run
  :param int num_points: number of time points produced.
  :param lmfit.Parameters: parameters to be estimated
  :param dict kwargs: optional arguments used in simulation
  :return list-lmfit.Parameters, list-float: parameters and RSQ from folds
  """
  df_obs = matrixToDF(obs_data)
  if num_points is None:
    num_points = len(df_obs)
  # Iterate for for folds
  fold_generator = foldGenerator(num_points, num_folds)
  result_parameters = []
  result_rsqs = []
  for train_indices, test_indices in fold_generator:
    # This function is defined inside the loop because it references a loop variable
    new_parameters = parameters.copy()
    fitted_parameters = fit(df_obs, method=method,
      indices=train_indices, parameters=new_parameters,
      sim_time=SIM_TIME, num_points=num_points,
      columns=columns,
      **kwargs)
    result_parameters.append(fitted_parameters)
    # Run the simulation using
    # the parameters estimated using the training data.
    test_estimates = runSimulation(sim_time=sim_time,
        num_points=num_points,
        parameters=fitted_parameters, **kwargs)
    df_est = matrixToDF(test_estimates)
    # Calculate RSQ
    rsq = calcRsq(df_obs, df_est, test_indices)
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
  result = np.reshape(residuals, (nrows, num_species))
  return result

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
