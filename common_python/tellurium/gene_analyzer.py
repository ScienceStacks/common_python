"""
Analyzes the suitability of a gene descriptor for a single gene.
The analysis decouples the model of the gene under analysis from
the rest of the gene network by using observational data instead
of model values in kinetics expressions. The analysis is done
in the context of the modeling game.

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
import pandas as pd
import numpy as np

PROTEIN_KINETICS = "a_protein%d*mRNA%d - d_protein%d*P%d"


class GeneAnalyzer(object):
  # Analyzes genes for a set of mRNA data

  def __init__(self, df_mrna):
    """
    :param pd.DataFrame df_mrna:
        columns: mRNA observations
        index: time
    :param str desc_stg: string descriptor for a gene
    Scope O.
    """
    self._stmt_initializations = util.readFile(
        cn.PATH_DICT[cn.INITIALIZATIONS])
    self._df_mrna = df_mrna
    # The following are used during analysis
    self.descriptor = None
    self.mrna_name = None
    self.reaction = None
    self.mrna_kinetics = None
    self.namespace = None
    # Results
    self.ser_est = None  # Estimate of the mRNA
    self.parameters = None  # Parameter values from estimates
    self.rsq = None  # rsq for estimate w.r.t. observations
     
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

  def do(self, desc_stg, end_time=1200):
    """
    Do an analysis for the descriptor string provided.
    :param str desc_stg:
    Updates:
      self.descriptor
      self.reaction
      self.mrna_name
      self.mrna_kinetics
    Scope OD.
    """
    # Initializations for a new gene descriptor
    self.descriptor = gn.GeneDescriptor.parse(desc_stg)
    network = gn.GeneNetwork()
    network.update([desc_stg])
    network.generate()
    self.reaction = gn.GeneReaction(descriptor)
    self.mrna_name = gn.GeneReaction.makeMrna(self.descriptor.ngene)
    reaction.generate()
    mrna_kinetics = self.__class__._makePythonExpression(
        self.reaction.mrna_kinetics)
    # Do estimates
    self._estimate(network.new_parameters, end_time)

  def _estimate(self, parameters, end_time):
    """
    For the current gene descriptor,
    estimates the time series for mRNA, protein, and fits parameter values.
    :param lmfit.Parameters parameters: Initial parameters values
    :param int end_time: end time of the integration
    Updates:
      self.ser_est
      self.parameters
    Scope OD.
    """
    def calcResiduals(parameters):
      """
      Calculates the residuals in a fit.
      :param lmfit.Parameters parameters:
      :return float: variance of residuals
      """
      self._calcMrnaEstimates(parameters)
      ser_res = self._df_mrna[self.mrna_name] - self.ser_est
      return ser_res.var()
    #
    self.parameters = copy.deepcopy(parameters)
    fitter = lmfit.Minimizer(calcResiduals, self.parameters)
    fitter_result = fitter.minimize(method="differential_evolution")
    # Assign the results
    self.params = fitter_result.parameters
    self.rsq = util.calcRsq(self._df_mrna[self.mrna_name],
        self.ser_est)

  def _calcMrnaEstimates(self, parameters):
    """
    Calculates mRNA estimates using numerical integration
    for a set of parameters.
    :param lmfit.Parameters parameters:
    Updates:
      self.ser_est
    Scope ODP.
    """
    self._initialize(parameters)
    # Construct initial values for the integration
    y0_arr = [self.namespace[gn.GeneReaction.makeProtein(n)]
        for n in range(1, gn.NUM_GENE+1)]
    y0_arr.insert(0, self.namespace[self.mrna_name])
    y0_arr = np.array(y0_arr)
    times = np.array(df_mrna.index.tolist())
    y_mat = odeint(self.__class__._calcKinetics, y0_arr, times, args=(self,))
    self.ser_est = pd.Series(y_mat[:, 0], index=times)
  
  def _initialize(self, parameters):
    """
    Initializes variables for a numerical integration using
    the parameters values.
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

  @staticmethod
  def _calcKinetics(y_arr, time, analyzer):
    """
    Evaluates the kinetics expressions for the mRNA under study (self.mrna_name) and
    all proteins given observed values of the mRNAs in self._df_mrna.
    :param np.array y_arr: y_arr[0] = mRNA, y_arr[1],...,y_arr[8] = P1...P8
    :param float time: time of the evaluation
    :param GeneAnalyzer analyzer:
    :return np.array: dydt for the elements of the vector
    Scope ODPT.
    """
    # FIXME: Generalize the selection of time indices
    # Adjust time
    time = 10*round(time/10)  # Adjust time to be consistent with the dataframe
    # Update the namespace with the new protein values and the
    # mRNA values from the observations
    for idx in range(1, 8):
      protein = gn.GeneReaction.makeProtein(idx)
      analyzer.namespace[protein] = y[idx]
      col = gn.GeneReaction.makeMrna(idx)
      analyzer.namespace[col] = analyzer.df_mrna.loc[time, col]
    # Update to the new values of mRNA for the gene under analysis
    analyzer.namespace[self.mrna_name] = y[0]
    # Calculate the drivatives
    dydt = [eval(analyzer.mrna_kinetics, analyzer.namespace)]
    for idx in range(1, 9):
      statement = PROTEIN_KINETICS % (idx, idx, idx, idx)
      dydt.append(eval(statement, analyzer.namespace))
    return dydt
