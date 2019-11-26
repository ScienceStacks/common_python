"""
Analyzes the suitability of a gene descriptor for a single gene.
The analysis decouples the model of the gene under analysis from
the rest of the gene network by using observational data instead
of model values in kinetics expressions. The analysis is done
in the context of the modeling game.
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



class GeneAnalyzer(object):

  def __init__(self, df_mrna, desc_stg):
    """
    :param pd.DataFrame df_mrna:
        columns: mRNA observations
        index: time
    :param str desc_stg: string descriptor for a gene
    """
    self._stmt_initializations = util.readFile(
        cn.PATH_DICT[cn.INITIALIZATIONS])
    descriptor = gn.GeneDescriptor.parse(desc_stg)
    self._gene_reaction = gn.GeneReaction(descriptor)
    self._gene_reaction.generate()
    self.network = gn.GeneNetwork()
    self.network.update([desc_stg])
    self.network.generate()
    # Public
    self.df_mrna = df_mrna
    self.mrna_name = gn.GeneReaction.makeMrna(descriptor.ngene)
    self.ngene = descriptor.ngene
    self.parameters_initial = self.network.new_parameters
    # Results
    self.ser_est = None  # Estimate of the mRNA
    self.parameters = copy.deepcopy(self.parameters_initial)
    # Other initializations
    self.initializeNamespace()  # self.namespace
    self.mrna_kinetics = self.__class__._transformKinetics(
        self._gene_reaction.mrna_kinetics)
    self.protein_kinetics = self.__class__._transformKinetics(
        "a_protein%d*mRNA%d - d_protein%d*P%d;")
     
  @staticmethod
  def _transformKinetics(kinetics):
    """
    Make the kinetics expressions python evaluatable.
    :param str kinetics: a tellurium kinetics expression
    :return str:
    """
    new_kinetics = kinetics.replace("^", "**")
    return new_kinetics.replace(";", "")
  
  def updateParameters(self, parameters):
    self.parameters = parameters
      
  def initializeNamespace(self):
    """
    Initializes the namespace to the values of the initial model
    and the parameters.
    """
    self.namespace = {}
    valuesdict = self.parameters.valuesdict()
    for name, value in valuesdict.items():
        self.namespace[name] = value
    exec(self._stmt_initializations, self.namespace)

  def do(self, desc_stg, end_time=1200):
    """
    Do an analysis for the descriptor string provided.
    :param str desc_stg:
    Updates
      self.descriptor
    """
    self.descriptor = gn.GeneDescriptor.parse(desc_stg)
    network = gn.GeneNetwork()
    network.update(desc_stg)
    network.generate()
    self._estimate(network.new_parameters, end_time)

  @staticmethod
  def calcKinetics(y_arr, time, analyzer):
    """
    Evaluates the kinetics expressions for the mRNA under study (self.mrna_name) and
    all proteins given observed values of the mRNAs in self.df_mrna.
    :param np.array y_arr: y_arr[0] = mRNA, y_arr[1],...,y_arr[8] = P1...P8
    :param float time: time of the evaluation
    :param GeneAnalyzer analyzer:
    :return float: variance of the residuals of the estimated and observed mRNA
    """
    # Adjust time
    time = 10*round(time/10)  # Adjust time to be consistent with the dataframe
    # Update the namespae 
    for idx in range(1, 8):
      protein = gn.GeneReaction.makeProtein(idx)
      analyzer.namespace[protein] = y[idx]
      col = gn.GeneReaction.makeMrna(idx)
      analyzer.namespace[col] = analyzer.df_mrna.loc[time, col]
    analyzer.namespace[self.mrna_name] = y[0]
    # Calculate the drivatives
    dydt = [eval(analyzer.mrna_kinetics, analyzer.namespace)]
    for idx in range(1, 9):
      statement = analyzer.protein_kinetics % (idx, idx, idx, idx)
      dydt.append(eval(statement, analyzer.namespace))
    return dydt

  def calcMrnaEstimates(self, parameters):
    """
    Calculates mRNA estimates using numerical integration.
    :param lmfit.Parameters parameters:
    :return pd.Series:
    """
    self.updateParameters(parameters)
    self.initializeNamespace()
    # Construct initial values
    y0_arr = [self.namespace[gn.GeneReaction.makeProtein(n)]
        for n in range(1, gn.NUM_GENE+1)]
    y0_arr.insert(0, self.namespace[self.mrna_name])
    y0_arr = np.array(y0_arr)
    times = np.array(df_mrna.index.tolist())
    y_mat = odeint(self.__class__.calcKinetics, y0_arr, times, args=(self,))
    ser = pd.Series(y_mat[:, 0], index=times)
    return ser

  def _estimate(self, end_time):
    """
    Estimates the time series for mRNA, protein, and fits
    parameter values.
    :param lmfit.Parameters parameters: Initial parameters values
    Updates:
      self.ser_mrna
      self.ser_protein
      self.parameters
    """
    def calcResiduals(parameters):
      """
      Calculates the residuals in a fit.
      :param lmfit.Parameters parameters:
      :return float: variance of residuals
      """
      ser_est = self.calcMrnaEstimates(parameters)
      ser_res = self.df_mrna[self.mrna_name] - ser_est
      return ser_res.var()
    #
    fitter = lmfit.Minimizer(calcResiduals, self.parameters)
    fitter_result = fitter.minimize(method="differential_evolution")
    self.params = fitter_result.parameters
    self.ser_est = self.calcMrnaEstimate(self.parameters)
    #
