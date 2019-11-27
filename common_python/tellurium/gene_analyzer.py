"""
Analyzes the suitability of a gene descriptor for a single gene.
The analysis decouples the model of the gene under analysis from
the rest of the gene network by using observational data instead
of model values in kinetics expressions. The analysis is done
in the context of the modeling game.

The methods are structure in the following hierarchy:
  1. Analyses for a given set of mRNA data (e.g., __init__)
  2. Analysis for a gene descriptor (e.g., do)
  3. Analysis for a set of parameter values for a gene descriptor
     (e.g., _estimate)
  4. Analysis for a time point for parameter values for a gene descriptor
     (e.g., _calcKinetics)
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
    """
    self._stmt_initializations = util.readFile(
        cn.PATH_DICT[cn.INITIALIZATIONS])
    self._df_mrna = df_mrna
    self._descriptor = None
    self._mrna_name = None
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
    """
    new_kinetics = kinetics.replace("^", "**")
    return new_kinetics.replace(";", "")
  
  def _initialize(self, parameters):
    """
    Initializes variables for a numerical integration using a set
    of parameters.
    Updates:
      self.parameters
      self._namespace
    """
    self.parameters = parameters
    # Base initializations
    self._namespace = {}
    exec(self._stmt_initializations, self._namespace)
    # Initializations for the parameters
    valuesdict = parameters.valuesdict()
    for name, value in valuesdict.items():
        self._namespace[name] = value

  def do(self, desc_stg, end_time=1200):
    """
    Do an analysis for the descriptor string provided.
    :param str desc_stg:
    Updates:
      self._descriptor
      self._reaction
      self._mrna_name
      self._mrna_kinetics
    """
    # Initializations for a new gene descriptor
    self._descriptor = gn.GeneDescriptor.parse(desc_stg)
    network = gn.GeneNetwork()
    network.update([desc_stg])
    network.generate()
    self._reaction = gn.GeneReaction(descriptor)
    self._mrna_name = gn.GeneReaction.makeMrna(self._descriptor.ngene)
    reaction.generate()
    self._mrna_kinetics = self.__class__._makePythonExpression(
        self._reaction.mrna_kinetics)
    # Do estimates
    self._estimate(network.new_parameters, end_time)

  @staticmethod
  def _calcKinetics(y_arr, time, analyzer):
    """
    Evaluates the kinetics expressions for the mRNA under study (self._mrna_name) and
    all proteins given observed values of the mRNAs in self._df_mrna.
    :param np.array y_arr: y_arr[0] = mRNA, y_arr[1],...,y_arr[8] = P1...P8
    :param float time: time of the evaluation
    :param GeneAnalyzer analyzer:
    :return np.array: dydt for the elements of the vector
    """
    # Adjust time
    time = 10*round(time/10)  # Adjust time to be consistent with the dataframe
    # Update the namespace with the new protein values
    for idx in range(1, 8):
      protein = gn.GeneReaction.makeProtein(idx)
      analyzer._namespace[protein] = y[idx]
      col = gn.GeneReaction.makeMrna(idx)
      analyzer._namespace[col] = analyzer.df_mrna.loc[time, col]
    # Update to the new values of mRNA
    analyzer._namespace[self._mrna_name] = y[0]
    # Calculate the drivatives
    dydt = [eval(analyzer.mrna_kinetics, analyzer._namespace)]
    for idx in range(1, 9):
      statement = PROTEIN_KINETICS % (idx, idx, idx, idx)
      dydt.append(eval(statement, analyzer._namespace))
    return dydt

  def _calcMrnaEstimates(self, parameters):
    """
    Calculates mRNA estimates using numerical integration
    for a set of parameters.
    :param lmfit.Parameters parameters:
    Updates:
      self.ser_est
    """
    self._initialize(parameters)
    # Construct initial values for the integration
    y0_arr = [self._namespace[gn.GeneReaction.makeProtein(n)]
        for n in range(1, gn.NUM_GENE+1)]
    y0_arr.insert(0, self._namespace[self._mrna_name])
    y0_arr = np.array(y0_arr)
    times = np.array(df_mrna.index.tolist())
    y_mat = odeint(self.__class__._calcKinetics, y0_arr, times, args=(self,))
    self.ser_ser = pd.Series(y_mat[:, 0], index=times)

  def _estimate(self, parameters, end_time):
    """
    For the current gene descriptor,
    estimates the time series for mRNA, protein, and fits parameter values.
    :param lmfit.Parameters parameters: Initial parameters values
    :param int end_time: end time of the integration
    Updates:
      self.ser_est
      self.parameters
    """
    def calcResiduals(parameters):
      """
      Calculates the residuals in a fit.
      :param lmfit.Parameters parameters:
      :return float: variance of residuals
      """
      self._calcMrnaEstimates(parameters)
      ser_res = self._df_mrna[self._mrna_name] - self.ser_est
      return ser_res.var()
    #
    self.parameters = copy.deepcopy(parameters)
    fitter = lmfit.Minimizer(calcResiduals, self.parameters)
    fitter_result = fitter.minimize(method="differential_evolution")
    # Assign the results
    self.params = fitter_result.parameters
    self.rsq = util.calcRsq(self._df_mrna[self._mrna_name],
        self.ser_est)
    #
