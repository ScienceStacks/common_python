'''Tests for model fitting.'''

from common_python.tellurium import model_fitting as mf

import lmfit
import numpy as np

############ CONSTANTS #############
IS_PLOT = False
NROWS = 10
NROWS_SUBSET = 5
NCOLS = 3
LENGTH = NROWS*NCOLS
INDICES = range(NROWS)
# Set to values used in mf.MODEL
TEST_PARAMETERS = lmfit.Parameters()
TEST_PARAMETERS.add('k1', value=0.1, min=0, max=10)
TEST_PARAMETERS.add('k2', value=0.2, min=0, max=10)

def makeData(nrows, ncols, valFunc=None):
  """
  Creates an array in the desired shape.
  :param int nrows:
  :param int ncols:
  :param Function value: Function
                      argument: number of values
                      return: iterable of values
  """
  length = nrows*ncols
  if valFunc is None:
    valFunc = lambda v: range(v)
  data = valFunc(length)
  matrix = np.reshape(data, (nrows, ncols))
  return matrix
  

def testReshapeData():
  data = makeData(NROWS, NCOLS)
  array = mf.reshapeData(data, range(NROWS_SUBSET))
  assert(len(array) == NROWS_SUBSET*NCOLS)
  assert(np.shape(array)[0] == len(array))

def testArrayDifference():
  matrix1 =  makeData(NROWS, NCOLS)
  matrix2 =  makeData(NROWS, NCOLS)
  array = mf.arrayDifference(matrix1, matrix2, INDICES)
  assert(sum(np.abs(array)) == 0)

def testCalcRsq():
  std = 0.5
  residuals = np.reshape(np.random.normal(0, std, LENGTH),
      (NROWS, NCOLS))
  matrix1 =  makeData(NROWS, NCOLS)
  matrix2 = matrix1 + residuals
  rsq = mf.calcRsq(matrix2, matrix1)
  var_est = (1 - rsq)*np.var(matrix1)
  var_exp = std*std
  assert(np.abs(var_est - var_exp) < 0.5)

def testMakeParameters():
  constants =  ['k1', 'k2', 'k3']
  parameters = mf.makeParameters(constants=constants)
  assert(len(parameters.valuesdict()) == len(constants))

def testMakeAverageParameters():
  """
  Constructs parameter values that are the average of existing parameters.
  """
  list_parameters = [TEST_PARAMETERS, TEST_PARAMETERS]
  average_parameters = mf.makeAverageParameters(
      list_parameters)
  test_dict = TEST_PARAMETERS.valuesdict()
  result_dict = average_parameters.valuesdict()
  for name in test_dict.keys():
    assert(test_dict[name] == result_dict[name])

def testRunSimulation():
  data1 = mf.runSimulation() 
  assert(data1[-1, 0] == mf.SIM_TIME)
  data2 = mf.runSimulation(
      parameters=TEST_PARAMETERS) 
  nrows, ncols = np.shape(data1)
  for i in range(nrows):
    for j in range(ncols):
      assert(np.isclose(data1[i,j], data2[i,j]))

def testPlotTimeSeries():
  # Smoke test only
  data = mf.runSimulation() 
  mf.plotTimeSeries(data, is_plot=IS_PLOT)
  mf.plotTimeSeries(data, is_scatter=True, is_plot=IS_PLOT)
  

def testMakeObservations():
  def test(num_points):
    obs_data = mf.makeObservations(
        num_points=num_points,
        road_runner=mf.ROAD_RUNNER)
    data = mf.runSimulation(
        num_points=num_points,
        road_runner=mf.ROAD_RUNNER)
    data = data[:, 1:]
    nrows, _ = np.shape(data)
    assert(nrows == num_points)
    std = np.sqrt(np.var(mf.arrayDifference(
        obs_data[:, 1:], data)))
    assert(std < 3*mf.NOISE_STD)
    assert(std > mf.NOISE_STD/3.0)
  test(mf.NUM_POINTS)
  test(2*mf.NUM_POINTS)

def testCalcSimulationResiduals():
  obs_data = mf.runSimulation(
      parameters=TEST_PARAMETERS)
  residuals = mf.calcSimulationResiduals(obs_data,
      TEST_PARAMETERS)
  assert(sum(residuals*residuals) == 0)

def testFit():
  obs_data = mf.makeObservations()
  parameters = mf.fit(obs_data)
  param_dict = dict(parameters.valuesdict())
  expected_param_dict = dict(mf.PARAMETERS.valuesdict())
  diff = set(param_dict.keys()).symmetric_difference(
      expected_param_dict.keys())
  assert(len(diff) == 0)

def testCrossValidate():
  obs_data = mf.makeObservations(
      parameters=TEST_PARAMETERS)
  results_parameters, results_rsqs = mf.crossValidate(
      obs_data)
  parameters_avg = mf.makeAverageParameters(
      results_parameters)
  params_dict = parameters_avg.valuesdict()
  for name in params_dict.keys():
    assert(np.abs(params_dict[name]  \
    - TEST_PARAMETERS.valuesdict()[name]) < 2*params_dict[name])

def testCrossValidate2():
  num_points = 20
  obs_data = mf.makeObservations(
      parameters=TEST_PARAMETERS, num_points=num_points)
  results_parameters, results_rsq = mf.crossValidate(
      obs_data, num_points=num_points, num_folds=10)
  parameters_avg = mf.makeAverageParameters(
      results_parameters)
  params_dict = parameters_avg.valuesdict()
  for name in params_dict.keys():
    assert(np.abs(params_dict[name]  \
    - TEST_PARAMETERS.valuesdict()[name]) < 2*params_dict[name])

def testMakeResidualsBySpecies():
  num_points = 20
  max_val = 10 
  residual_matrix = _getResiduals(num_points)
  assert(np.shape(residual_matrix)[0] == num_points)
  assert(sum(sum(residual_matrix)) < max_val)

def testMakeSyntheticObservations():
  num_points = 20
  kwargs = {'model': mf.MODEL,
            'num_points': num_points,
           }
  residual_matrix = _getResiduals(num_points, model=kwargs['model'])
  syn_data = mf.makeSyntheticObservations(residual_matrix, **kwargs)
  assert(np.shape(syn_data)[0] == num_points)

def _makeParameterList(num_points, count, **kwargs):
  residual_matrix = _getResiduals(num_points, model=kwargs['model'])
  return mf.doBootstrapWithResiduals(residual_matrix, count=count,
      **kwargs)

def testDoBootstrapWithResiduals():
  num_points = 20
  count = 3
  list_parameters = _makeParameterList(num_points, count,
      model=mf.MODEL)
  assert(len(list_parameters) == count)
  for parameters in list_parameters:
    assert(isinstance(parameters, lmfit.Parameters))

def _getResiduals(num_points, model=mf.MODEL):
  obs_data = mf.makeObservations(model=model,
      parameters=TEST_PARAMETERS, num_points=num_points)
  return mf.makeResidualsMatrix(obs_data, model,
      TEST_PARAMETERS, num_points=num_points)

def testDoBootstrap():
  num_points = 20
  count = 3
  model = mf.MODEL
  obs_data = mf.makeObservations(model=model,
      parameters=TEST_PARAMETERS, num_points=num_points)
  confidence_dict = mf.doBootstrap(obs_data, model,
      TEST_PARAMETERS, count=count,
      num_points=num_points)
  params_dict = TEST_PARAMETERS.valuesdict()
  diff = set(params_dict.keys()).symmetric_difference(
      confidence_dict.keys())
  assert(len(diff) == 0)
  for value in confidence_dict.values():
    assert(len(value) == 2)

def testDoBootstrap2():
  model0 = """
       # True model
       A  -> B + D; k1*A
       B -> D; k2*B
       D -> C; k3*A*B
        
       A = 5;
       B = 0;
       C = 0;
       D = 0;
       k1 = 0.08
       k2 = 0.1
       k3 = 0.1
  """
  num_points = 20
  sim_time = 20
  unfitted_parameters = mf.makeParameters(
      constants=['k1', 'k2', 'k3'])
  full_obs_data = mf.makeObservations(model=model0, 
      noise_std=0.3, num_points=num_points, sim_time=sim_time)
  result = mf.doBootstrap(full_obs_data, 
      model=model0, parameters=unfitted_parameters, 
      num_points=num_points, sim_time=sim_time, count=5)
  
 
def testMakeParameterStatistics():
  num_points = 20
  count = 3
  list_parameters = _makeParameterList(num_points, count,
      model=mf.MODEL)
  def test(confidence_limits):
    statistics = mf.makeParameterStatistics(list_parameters,
        confidence_limits)
    for key in statistics.keys():
      assert(len(statistics[key]) == 2)
  #
  test((5, 95))
  test(None)
  
   
  
if __name__ == '__main__':
  testDoBootstrap2()
  if True:
    testReshapeData() 
    testArrayDifference() 
    testCalcRsq()
    testMakeParameters()
    testMakeAverageParameters()
    testRunSimulation()
    testPlotTimeSeries()
    testCalcSimulationResiduals()
    testFit()
    testCrossValidate()
    testMakeObservations()
    testCrossValidate2()
    testMakeResidualsBySpecies()
    testMakeSyntheticObservations()
    testDoBootstrapWithResiduals()
    testDoBootstrap()
    testMakeParameterStatistics()
  print("OK")
