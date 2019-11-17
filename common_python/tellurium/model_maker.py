"""
Construct a Model for the Modeling Game.

GeneMaker creates the reaction string for an mRNA and
identifies the constants that must be estimated.

A gene is described by a descriptor string. There are
three cases.
1. 0 TF: g
2. 1 TF: gsp
3. 2 TF: gspisp

where: 
  g is the gene number
  i indicates the kind of integration: A for and O for or
  s is either "+" or "-" to indicate that the protein activates
    or inhibits the gene product
  p is a protein number
"""

import model_fitting as mf
import modeling_game as mg

from collections import namedtuple
import copy
import numpy as np
import os
import pandas as pd

FILE_HEAD = "model_head.txt"
FILE_TAIL = "model_tail.txt"
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_HEAD = os.path.join(FILE_HEAD, CUR_DIR)
PATH_TAIL = os.path.join(FILE_TAIL, CUR_DIR)
NGENE = 8  # Number of genes
# Structure of gene string description
POS_GENE = 0
POS_SIGN_1 = 1
POS_PROTEIN_1 = 2

GeneDescriptor = namedtuple("GeneDescriptor",
    "ngene is_or_integration nprots is_activates")
GeneSpecification = namedtuple("GeneSpecification",
    "reaction constants")

# Initial arcs in reaction network
INITIAL_NETWORK = [
    ]
P0 = "INPUT"  # Protein 0 is the input
IDX_L = 0
IDX_DMRNA = 1


class GeneMaker(object):
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
    self._k_index = 0  # Index of the K constants
    self._H = self._makeVar("H")  # H constant
    self._Vm = self._makeVar("Vm")  # H constant
    self._mrna = self._makeVar("mRNA")
    # The following are produced when a reaction is constructed
    self.constants = [
        self._makeVar("L"),       # IDX_L
        self._makeVar("d_mRNA"),  # IDX_DMRNA
        ]
    self.reaction = None
      
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
    var = "K%d_%d" % (self._k_index, self._ngene)
    self.constants.append(var)
    return var

  def _makePVar(self, nprot):
    if nprot == 0:
      stg = "INPUT"
    else:
      stg = "P%d" % nprot
    return stg

  def _makeBasicKinetics(self):
    return "%s - %s*%s" % (
        self.constants[IDX_L], 
        self.constants[IDX_DMRNA],
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
    sep = ""
    if self._is_or_integration:
      is_first = True
      for is_activate, term  in zip(self._is_activates, terms[:-1]):
        if is_first:
          is_first = False
        else:
          sep = " + "
        if is_activate:
          numerator += "%s %s" %  (sep, term)
      # Include the joint occurrence of the TFs
      if all(self._is_activates):
        numerator += "%s %s" %  (sep, terms[-1])
      # If one is a repressor, then include state of both missing
      else:
        numerator += "%s 1" %  sep
      if len(numerator) == 0:
        numerator = "1"  # Ensure has at least one term
    else:  # AND integration
      numerator = "1"
      if all(self._is_activates):
        numerator = terms[-1]
      elif self._is_activates[0]:
        numerator = terms[0]
      else:
        if len(self._is_activates) > 1:
          if self._is_activates[1]:
            numerator = terms[1]
    return "%s * ( %s ) / ( %s )" % (
        self._Vm, numerator, denominator)
  
  def _makeKinetics(self):
    if len(self._nprots) == 0:
      stg = self._makeBasicKinetics()
    else:
      stg = "%s + %s" % (self._makeBasicKinetics(),
          self._makeTFKinetics())
      self.constants.extend([
          self._Vm,
          self._H,
          ])
    return stg

  def makeReaction(self):
    """
    Updates the reaction string, self.reaction.
    """
    label = self._makeVar("J")
    self.reaction = "%s: => %s; %s" % (label, 
        self._mrna, self._makeKinetics())

  @staticmethod
  def _parseDescriptorString(string):
    """
    Parses a descriptor string (as described in the module
    comments).
    :param str string: descriptor string
    :return GeneDescriptor:
    """
    # Initializations
    string = str(string)  # With 0 TFs, may have an int
    nprots = []
    is_activates = []
    is_or_integration = True
    #
    def extractTF(stg):
      if stg[0] == "+":
        is_activate = True
      else:
        is_activate = False
      nprots.append(int(stg[1]))
      is_activates.append(is_activate)     
    # Extract gene
    ngene = int(string[0])
    #
    if len(string) >= 3:
      extractTF(string[1:3])
    if len(string) == 6:
      if string[4] == "O":
        is_or_integration = True
      else:
        is_or_integration = False
      extractTF(string[4:6])
    #
    return GeneDescriptor(
        ngene=ngene,
        is_or_integration=is_or_integration,
        nprots=nprots,
        is_activates=is_activates)
    
  def __str__(self):
    if self.reaction is None:
      self.makeReaction()
    return self.reaction

  @classmethod
  def do(cls, string):
    """
    Constructs the reaction for the gene.
    :param str string: String representation of a gene description
    :return GeneMaker:
    """
    descriptor = cls._parseDescriptorString(string)
    maker = GeneMaker(descriptor.ngene, descriptor.is_or_integration)
    for nprot, is_activate in  \
        zip(descriptor.nprots, descriptor.is_activates):
      maker.addProtein(nprot, is_activate)
    maker.makeReaction()
    return maker
      

#TODO: Implement the code
class ModelMaker(object):
  # Create a full model.

  def __init__(self):
    """
    :param list-str gene_descriptions: TF descriptions for genes
    """
    self._ngene = NGENE
    self._constants = []  # Constants in the model
    self.parameters = None  # lmfit.Parameters for model
    self.model = None  # Model string
    self._description_dict = {n: 
        ModelMaker._makeDefaultDescription(n)
        for n in self._getGeneNumbers()}

  def _getGeneNumbers(self):
    return [n for n in range(1, self._ngene+1)] 

  @staticmethod
  def _makeDefaultDescription(ngene):
    return "%dX" % ngene

  def addDescriptions(self, descriptions, is_set_constants=True):
    """
    Cumulative adds to the model and the parameters.
    self._descriptions = gene_descriptions
    :param bool is_set_constants: initialize constants to 0
    """
    pass

  def make(self):
    """
    Generates a model
    Updates
      self.parameters
      self.model
    """
    model_str = ""
    # 1: Append head
    # 2: Construct gene reactions and append
    # 3: Append tail
    # 4: Construct list of constants and parameters
    pass

  def _addGene(self, ngene):
    """
    Uses the gene descriptions. If none present, generates
    a "null model"
    """
    pass

  def copy(self):
    """
    Copies the GeneMaker.
    :return GeneMaker:
    """
    return copy.deepcopy(self)

  def makeProteinConstants(self):
    """
    :return list-str: list of constants for protein reactions
    """
    pass

  @staticmethod
  def _readFile(path):
    with open(path) as fd:
      result = fd.readlines(path)
    return "\n".join(result)

  def __str__(self):
    return self._model
