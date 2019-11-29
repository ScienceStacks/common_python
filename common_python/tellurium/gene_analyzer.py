"""
Analyzes the suitability of a gene descriptor for a single gene.
The analysis decouples the model of the gene under analysis from
the rest of the gene network by using observational data instead
of model values in kinetics expressions. The analysis is done
in the context of the modeling game.

The analysis does not fit the following parameters:
  a_protein, d_protein

The methods have the following progressively more restrictive scopes:
  Scope O: Same observational data
           (an instance)
  Scope OD: Same obsevational data and gene descriptor 
            (invocation of do)
  Scope ODP: Same observational data, gene descriptor, and parameter values
             (invocation of self._calcMrnaEstimates)
  Scope ODPT: Same observational data, gene descriptor, parameter values, and time
              (invocation of _calcKinetics)
The rule is that a narrow scope (e.g., ODP) cannot call a broad scope (e.g., OD).
"""

import constants as cn
import gene_network as gn
import model_fitting as mf
import modeling_game as mg
import util

import copy
import lmfit   # Fitting lib
import numpy as np
import os
import pandas as pd
import scipy

PROTEIN_KINETICS = "a_protein%d*mRNA%d - d_protein%d*P%d"
DEFAULT_CONSTANT = "Vm1"
NUM_TO_TIME = 10


class GeneAnalyzer(object):
  # Analyzes genes for a set of mRNA data

  def __init__(self, mrna_path=cn.MRNA_PATH):
    """
    :param pd.DataFrame df_mrna:
        columns: mRNA observations
        index: time
    :param str desc_stg: string descriptor for a gene
    Scope O.
    """
    # The folowing are in the D scope
    self._stmt_initializations = util.readFile(
        cn.PATH_DICT[cn.INITIALIZATIONS])
    self._mrna_path = mrna_path
    self._df_mrna = pd.read_csv(mrna_path)
    self._df_mrna = self._df_mrna.set_index(cn.TIME)
    self._getProteinDF()  # self._df_protein
    # The following are in the OD Scope.
    self.network = None
    self.descriptor = None
    self.mrna_name = None
    self.reaction = None
    self.mrna_kinetics = None
    self.mrna_kinetics_compiled = None  # Compiled mRNA kinetics
    self.end_time = None
    # The following are in the ODP Scope
    self.namespace = None
    # Results
    self.ser_est = None  # Estimate of the mRNA
    self.parameters = None # Parameter values from estimates
    self.rsq = None  # rsq for estimate w.r.t. observations

  def _getProteinDF(self):
    """
    Obtains the protein dataframe from a file, it exists.
    Updates
      self._df_protein - columns are protein; index is time
    """
    path = GeneAnalyzer._makeProteinPath(mrna_path=self._mrna_path)
    if os.path.isfile(path):
      self._df_protein = pd.read_csv(path)
      self._df_protein = self._df_protein.set_index(cn.TIME)
      return
    else:
      raise ValueError("Cannot find generated protein file %s"
          % path)

  @staticmethod
  def _makeProteinPath(mrna_path=cn.MRNA_PATH):
    """
    Creates the path to the protein CSV file based on 
    the path to the mRNA file.
    """
    path = mrna_path.replace(".csv", "")
    return "%s_protein.csv" % path

  @classmethod
  def makeProteinDF(cls, mrna_path=cn.MRNA_PATH,
     is_write=True, end_time=1200):
    """
    Creates a CSV file with the estimates of protein concentrations.
    :param str mrna_path: path to mRNA data
    :param int end_time: upper end of integration
    :return pd.DataFrame:
    Creates a file in the same directory as cn.MRNA_PATH with
    "_protein.csv".
    This method runs for a long time.
    """
    # Initializations
    df_mrna, compileds = cls.proteinInitializations(mrna_path)
    # Integrate to obtain protein concetrations
    df_sub = df_mrna[df_mrna.index <= end_time]
    times = df_sub.index.tolist()
    y0_arr = np.repeat(0, gn.NUM_GENE)
    y_mat = scipy.integrate.odeint(GeneAnalyzer._proteinDydt,
         y0_arr, times,
         args=(df_mrna, compileds))
    # Structure the dataframe
    columns = [gn.GeneReaction.makeProtein(n)
        for n in range(1, gn.NUM_GENE + 1)]
    df = pd.DataFrame(y_mat)
    df.columns = columns
    df.index = times
    df.index.name = cn.TIME
    df = df.reset_index()
    # Write the results
    if is_write:
      protein_path = GeneAnalyzer._makeProteinPath(mrna_path)
      df.to_csv(protein_path, index=False)
    return df

  @staticmethod
  def proteinInitializations(mrna_path):
    """
    Constructs data needed to initialize the estimates of proteins
    :return pd.DataFrame, list-Bytecodes: df_mrna, compileds
    """
    path = GeneAnalyzer._makeProteinPath(mrna_path)
    df_mrna = pd.read_csv(mrna_path)
    df_mrna = df_mrna.set_index(cn.TIME)
    # Construct the compiled expressions for protein kinetics
    compileds = []
    for idx in range(1, gn.NUM_GENE + 1):
      expression = PROTEIN_KINETICS % (idx, idx, idx, idx)
      compileds.append(compile(
        expression, 'protein%d_kinetics' % idx, 'eval'))
    return df_mrna, compileds

  @staticmethod
  def _proteinDydt(y_arr, time, df_mrna, compileds):
    """
    Evaluates the kinetics expressions for the mRNA under study
    (self.mrna_name) and its associated protein (self.protein_name)
    :param np.array y_arr: y_arr[0],...y[NUM_GENE-1] = P1, ..., P8
    :param float time: time of the evaluation
    :param pd.DataFrame df_mrna: observed mRNA (columns); index is time
    :param list-Bytecodes compileds: compiled expressions for protein kinetics
    :return np.array: dydt for P1, ..., P8
    """
    def adj(n):
      # Adjusts the index
      return n + 1
    #
    time = GeneAnalyzer._adjustTime(time)
    namespace = {}
    dydts = []
    stmt_initializations = util.readFile(
        cn.PATH_DICT[cn.INITIALIZATIONS])
    exec(stmt_initializations, namespace)
    # Update the namespace and calculate the derivative
    for idx in range(gn.NUM_GENE):
      col = gn.GeneReaction.makeMrna(adj(idx))
      namespace[col] = df_mrna.loc[time, col]
      protein = gn.GeneReaction.makeProtein(adj(idx))
      namespace[protein] = y_arr[idx]
      dydt = eval(compileds[idx], namespace)
      dydts.append(dydt)
    # 
    return dydts

  def do(self, desc_stg, end_time=1200):
    """
    Do an analysis for the descriptor string provided.
    :param str desc_stg:
    Updates:
      self.ser_est
      self.parameters
    Scope: OD.
    """
    def calcResiduals(parameters):
      """
      Calculates the residuals in a fit.
      :param lmfit.Parameters parameters:
      :return list-float: variance of residuals
      Scope: ODP.
      """
      self._calcMrnaEstimates(parameters)
      ser_obs = self._df_mrna.loc[:self.end_time, self.mrna_name]
      del ser_obs[end_time]
      ser_res = ser_obs - self.ser_est
      return ser_res.to_list()
    # Initializations
    self._initializeODScope(desc_stg, end_time)
    # Do the fits
    fitter = lmfit.Minimizer(calcResiduals, self.parameters)
    fitter_result = fitter.minimize(method="differential_evolution")
    # Assign the results
    self.params = fitter_result.params
    self.rsq = util.calcRsq(self._df_mrna[self.mrna_name],
        self.ser_est)

  def _initializeODScope(self, desc_stg, end_time):
    """
    Initializes instance variables for the OD Scope.
    :param str desc_stg:
    :param int end_time: ending time of simulation
    Scope: OD.
    """
    self.end_time = end_time
    self.descriptor = gn.GeneDescriptor.parse(desc_stg)
    self.mrna_name = gn.GeneReaction.makeMrna(self.descriptor.ngene)
    self.network = gn.GeneNetwork()
    self.network.update([desc_stg])
    self.network.generate()
    self.reaction = gn.GeneReaction(self.descriptor)
    self.reaction.generate()
    self.mrna_kinetics = GeneAnalyzer._makePythonExpression(
        self.reaction.mrna_kinetics)
    self.parameters = copy.deepcopy(self.network.new_parameters)
    if not (isinstance(self.parameters, lmfit.Parameters)):
      self.parameters = mg.makeParameters([DEFAULT_CONSTANT])
    elif len(self.parameters.valuesdict()) == 0:
      self.parameters = mg.makeParameters([DEFAULT_CONSTANT])
    # Compile kinetics
    self.mrna_kinetics_compiled = compile(
        self.mrna_kinetics, 'mrna_kinetics', 'eval')
     
  @staticmethod
  def _makePythonExpression(kinetics):
    """
    Transforms the kinetics expressions into python.
    :param str kinetics: a tellurium kinetics expression
    :return str:
    No scope constraints.
    """
    new_kinetics = kinetics.replace("^", "**")
    return new_kinetics.replace(";", "")

  def _calcMrnaEstimates(self, parameters):
    """
    Calculates mRNA estimates using numerical integration
    for a set of parameters.
    :param lmfit.Parameters parameters:
    Updates:
      self.ser_est
    Scope ODP.
    """
    self._initializeODPScope(parameters)
    # Construct initial values for the integration
    if False:
      y0_arr = [self.namespace[gn.GeneReaction.makeProtein(n)]
        for n in range(1, gn.NUM_GENE+1)]
    else:
      y0_arr = []
    y0_arr.insert(0, self.namespace[self.mrna_name])
    y0_arr = np.array(y0_arr)
    times = np.array(self._df_mrna.index.tolist())
    num_times = int(self.end_time/NUM_TO_TIME)
    times = times[0:num_times]
    y_mat = scipy.integrate.odeint(GeneAnalyzer._calcKinetics,
        y0_arr, times, args=(self,))
    self.ser_est = pd.Series(y_mat[:, 0], index=times)

  def eulerOdeint(self, func, y0_arr, times, num_iter=10):
    """
    Does Euler integration for a specified number of iterations.
    :param Function func:
    :param np.array y0_arr:
    :param list-float times: first time must be 0
    :param int num_iter: number of iterations for a point
    :return np.array: matrix with indices corresponding to times
    """
    values = [np.array([float(v) for v in y0_arr])]
    for idx, time in enumerate(times[1:]):
      y_arr = np.array(values[idx])
      new_y_arr = np.array(y_arr)
      last_time = times[idx]
      incr = (time - last_time)/num_iter
      for count in range(num_iter):
        cur_time = last_time + (count+1)*incr
        dydt = func(y_arr, cur_time, self)
        new_y_arr = new_y_arr + np.array(dydt)*incr
      values.append(np.array(new_y_arr))
    return np.array(values)
  
  def _initializeODPScope(self, parameters):
    """
    Initializes variables for the ODP scope.
    Updates:
      self.parameters
      self.namespace
    Scope ODP.
    """
    self.parameters = parameters
    # Base initializations
    self.namespace = {}
    exec(self._stmt_initializations, self.namespace)
    # Initializations for the parameters
    # This may overwrite vales in self._stmt_initialization
    valuesdict = parameters.valuesdict()
    for name, value in valuesdict.items():
        self.namespace[name] = value

  # FIXME: Generalize the selection of time indices
  @staticmethod
  def _adjustTime(time):
    return NUM_TO_TIME*round(time/NUM_TO_TIME)

  @staticmethod
  def _calcKinetics(y_arr, time, analyzer):
    """
    Evaluates the kinetics expressions for the mRNA under study
    (self.mrna_name) and its associated protein (self.protein_name)
    :param np.array y_arr: y_arr[0] = mRNA
    :param float time: time of the evaluation
    :param GeneAnalyzer analyzer:
    :return np.array: dydt for the elements of the vector
    Scope ODPT.
    Just estimate mRNA and its protein.
    """
    # Adjust time
    time = GeneAnalyzer._adjustTime(time)
    # Update the namespace for the current time and mRNA
    analyzer.namespace[analyzer.mrna_name] = y_arr[0]
    for idx in range(1, gn.NUM_GENE + 1):
      col = gn.GeneReaction.makeProtein(idx)
      try:
        analyzer.namespace[col] = analyzer._df_protein.loc[time, col]
      except:
        import pdb; pdb.set_trace()
    # Evaluate the derivative
    dydt = [eval(analyzer.mrna_kinetics_compiled, analyzer.namespace)]
    return dydt
  

if __name__ == '__main__':
  GeneAnalyzer.makeProteinDF()
