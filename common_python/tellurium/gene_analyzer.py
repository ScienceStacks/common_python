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

import lmfit   # Fitting lib
import pandas as pd
import numpy as np



class GeneAnalyzer(Object):

  def __init__(self, ngene, df_obs):
    """
    :param int ngene: number of the gene
    :param pd.DataFrame df_obs: observational data
        columns: species names
        index: time
    """
    self._ngene = ngene
    self._df_obs = df_obs
    self._times = self._df_obs.index.tolist()
    self._variables = self._df_obs.columns.tolist()
    self._namespace = {}  # Name space for evaluating kinetics
    self._initializeNamespace()
    # Public
    self.descriptor = None  # Descriptor which estimate is done
    self.ser_mrna = None  # Estimated values of mrna
    self.ser_protein = None  # Estimated values of protein
    self.parameters = None  # Providers constructed from the estimate

  def _initializeNamespace(self):
    """
    Creates evaluating kinetics expressions
    """
    statements = util.readFile(cn.PATH_DICT[cn.INITIALIZATIONS])
    exec(statements, self._namespace)
    
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

  def _estimate(self, parameters, end_time):
    """
    Estimates the time series for mRNA, protein, and fits
    parameter values.
    :param lmfit.Parameters parameters: Initial parameters values
    Updates:
      self.ser_mrna
      self.ser_protein
      self.parameters
    """
    fitter = lmfit.Minimizer(integ, parameters)
    fitter_result = fitter.minimize(method="differential_evolution")
    #
    def
