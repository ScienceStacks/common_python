"""Construct a Model for the Modeling Game."""

import model_fitting as mf
import modeling_game as mg

from collections import namedtuple
import numpy as np
import pandas as pd

TranscriptionFactor = namedtuple("TranscriptionFactor", "nprot activate")

# Initial arcs in reaction network
INITIAL_NETWORK = [
    ]
P0 = "INPUT"  # Protein 0 is the input
IDX_L = 0
IDX_DMRNA = 1

# Modified model
RNA1 = '''
  J1:  => mRNA1; L1 + Vm1*((K1_1*INPUT^H1 + K2_1*P4^H1 + K1_1*K3_1*INPUT^H1*P4^H1)/(1 + K1_1*INPUT^H1 + K2_1*P4^H1 + K1_1*K3_1*INPUT^H1*P4^H1)) - d_mRNA1*mRNA1;
// Created by libAntimony v3.9.4
model *pathway()

  // Compartments and Species:
  species INPUT, P1, mRNA1, P2, mRNA2, P3, mRNA3, P4, mRNA4, P5, mRNA5, P6;
  species mRNA6, P7, mRNA7, P8, mRNA8;
  
  J1:  => mRNA1; L1 + Vm1*((K1_1*INPUT^H1 + K2_1*P4^H1 + K1_1*K3_1*INPUT^H1*P4^H1)/(1 + K1_1*INPUT^H1 + K2_1*P4^H1 + K1_1*K3_1*INPUT^H1*P4^H1)) - d_mRNA1*mRNA1;
  F1:  => P1; a_protein1*mRNA1 - d_protein1*P1;
  J2:  => mRNA2; L2 + Vm2*(K1_2*P4^H2/(1 + K1_2*P4^H2)) - d_mRNA2*mRNA2;
  F2:  => P2; a_protein2*mRNA2 - d_protein2*P2;
  J3:  => mRNA3; L3 + Vm3*(K1_3*P6^H3/(1 + K1_3*P6^H3)) - d_mRNA3*mRNA3;
  F3:  => P3; a_protein3*mRNA3 - d_protein3*P3;
  J4:  => mRNA4; L4 + Vm4*(1/(1 + K1_4*P2^H4)) - d_mRNA4*mRNA4;
  F4:  => P4; a_protein4*mRNA4 - d_protein4*P4;
  J5:  => mRNA5; L5 - d_mRNA5*mRNA5;
  F5:  => P5; a_protein5*mRNA5 - d_protein5*P5;
  J6:  => mRNA6; L6 + Vm6*(K1_6*P7^H6/(1 + K1_6*P7^H6 + K2_6*P1^H6 + K1_6*K2_6*P7^H6*P1^H6)) - d_mRNA6*mRNA6;
  F6:  => P6; a_protein6*mRNA6 - d_protein6*P6;
  J7:  => mRNA7; L7 +  Vm7*( K1_7*P1^H7/(1 + K1_7*P1^H7) + 1/(1 + K2_7*P7^H7)) - d_mRNA7*mRNA7;
  F7:  => P7; a_protein7*mRNA7  - d_protein7*P7;
  J8:  => mRNA8; L8 + Vm8*(1/(1 + K1_8*P1^H8)) - d_mRNA8*mRNA8;
  F8:  => P8; a_protein8*mRNA8 - d_protein8*P8;
  '''


class ReactionMaker(object):
  """Creates the reaction for gene production of mRNA."""
    
  def __init__(self, ngene, is_or_integration=True):
    """
    :param int ngene: number of the gene
    :param bool is_or_integration: logic used to combine terms
    """
    self._ngene = ngene
    self._is_or_integration = is_or_integration
    self._nprots = []
    self._is_activates = []
    self._constants = [
        self._makeVar("L"),       # IDX_L
        self._makeVar("d_mRNA"),  # IDX_DMRNA
        ]
    self._k_index = 0  # Index of the K constants
    self._H = self._makeVar("H")  # H constant
    self._Vm = self._makeVar("Vm")  # H constant
    self._mrna = self._makeVar("mRNA")
    self._reaction = None
      
  def addProtein(self, nprot, is_activate=True):
    """
    :param int nprot: numbers of the protein that is TF for the gene
    :param list-bool is_activation: whether the gene is activation (True)
    """
    self._nprots.append(nprot)
    self._is_activates.append(is_activate)
      
  def _makeVar(self, name):
    return "%s%d" % (name, self._ngene)

  def _makeKVar(self):
    self._k_index += 1
    var = "K%d_%d" % (self._ngene, self._k_index)
    self._constants.append(var)
    return var

  def _makePVar(self, nprot):
    return "P%d" % nprot

  def _makeBasicKinetics(self):
    return "%s - %s*%s" % (
        self._constants[IDX_L], 
        self._constants[IDX_DMRNA],
        self._mrna)

  def _makeTerm(self, nprots):
    """
    Creates the term Km_n*(Pi*Pj)^Hm
    :param list-int nprots:
    """
    term = "%s" % self._makeKVar()
    for nprot in nprots:
      term = term + "*%s^%s" % (self._makePVar(nprot), self._H)
    return term

  def _makeTFKinetics(self):
    """
    Creates the kinetics for the transcription factors
    """
    terms = [self._makeTerm([p]) for p in self._nprots]
    if len(self._nprots) > 1:
      terms.append(self._makeTerm(self._nprots))
    denominator = "1"
    for term in terms:
      denominator += " + %s" % term
    numerator = ""
    if self._is_or_integration:
      is_first = True
      for is_activate, term  in zip(self._is_activates, terms[:-1]):
        if is_first:
          sep = ""
          is_first = False
        else:
          sep = " + "
        if is_activate:
          numerator += "%s %s" %  (sep, term)
      if len(numerator) == 0:
        numerator = "1"  # Ensure has at least one term
    else:  # AND integration
      numerator = self._makeTerm(self._nprots)
      if all(self._is_activates):
        numerator = terms[-1]
      elif self._is_activates[0]:
        numerator = terms[0]
      elif self._is_activates[1]:
        numerator = terms[1]
    return "%s * ( %s ) / ( %s )" % (
        self._Vm, numerator, denominator)
  
  def _makeKinetics(self):
    if len(self._nprots) == 0:
      stg = self._makeBasicKinetics
    else:
      stg = "%s + %s" % (self._makeBasicKinetics(),
          self._makeTFKinetics())
      self._constants.extend([
          self._Vm,
          self._H,
          ])
    return stg

  def makeReaction(self):
    if self._reaction is None:
      label = self._makeVar("J")
      self._reaction = "%s: => %s; %s" % (label, 
          self._mrna, self._makeKinetics())
    return self._reaction
    
  def __str__(self):
    return self._reaction
      

class ModelMaker(object):

  def __init__(self, num_mrna=8):
    self._rna_productions = []
    self._parameters = None
    self._model = ""

  def addGene(ngene, nprots, is_or_integration=True):
    pass

  def __str__(self):
    return self._model

