"""
Constructs a gene network for the modeling game.

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

The module is implemented as follows:

  GeneDescriptor implements a representation of the
  transcription factors (proteins) that activate/inhibit a gene.

  GeneReaction constructions the mRNA reaction for a gene,
  along with providing the constants in the reaction.

  GeneNetwork generates the entire gene network for the game.
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
# Initial network from the modeling game
INITIAL_NETWORK = ["1+0O+4", "2+4", "3+6", "4-2", "6+7A-1", "8-1"]
# Structure of gene string descriptor
POS_GENE = 0
POS_SIGN_1 = 1
POS_PROTEIN_1 = 2

# Initial arcs in reaction network
INITIAL_NETWORK = [
    ]
P0 = "INPUT"  # Protein 0 is the input
IDX_L = 0
IDX_DMRNA = 1


################ Helper Functions ###################
def _setList(values):
  if values is None:
    return []
  else:
    return values

def _equals(list1, list2):
  diff = set(list1).symmetric_difference(list2)
  return len(diff) == 0


################ Classes ###################
class GeneDescriptor(object):
  # Describes the transcription factors (protein) for a gene
  # and how they affect the gene's activation

  def __init__(self, ngene, nprots=None, 
      is_activates=None, is_or_integration=True):
    """
    :param int ngene: number of the gene
    :param list-int nprots: list of protein TFs
    :param list-bool is_activates: list of how corresponding prot impacts gene
    :param bool is_or_integration: logic used to combine terms
    """
    self.ngene = int(ngene)
    self.nprots = [int(p) for p in _setList(nprots)]
    self.is_activates = _setList(is_activates)
    self.is_or_integration = is_or_integration

  def equals(self, other):
    result = True
    result = result and (self.ngene == other.ngene)
    result = result and _equals(self.nprots, other.nprots)
    result = result and _equals(self.is_activates, other.is_activates)
    result = result and (
        self.is_or_integration == other.is_or_integration)
    return result

  def __str__(self):
    def makeTerm(is_activate, nprot):
      if is_activate:
        sign = "+"
      else:
        sign = "-"
      stg = "%s%d" % (sign, nprot)
      return stg
    #
    stg = str(self.ngene)
    if len(self.nprots) > 0:
      stg += makeTerm(self.is_activates[0], self.nprots[0])
    if len(self.nprots) == 2:
      if self.is_or_integration:
        conjunction = "O"
      else:
        conjunction = "A"
      stg += conjunction
      stg += makeTerm(self.is_activates[1], self.nprots[1])
    return stg
      

  @classmethod
  def parse(cls, string):
    """
    Parses a descriptor string (as described in the module
    comments).
    :param str string: gene descriptor string
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
        ngene,
        is_or_integration=is_or_integration,
        nprots=nprots,
        is_activates=is_activates)


######################################################
class GeneReaction(object):
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
    
  def __str__(self):
    if self.reaction is None:
      self.makeReaction()
    return self.reaction

  @classmethod
  def do(cls, descriptor):
    """
    Constructs the reaction and constants for the gene.
    :param str/GeneDescriptor descriptor: gene descriptor
    :return GeneReaction:
    """
    if isinstance(desc, str):
      descriptor = GeneDescriptor.parse(string)
    maker = GeneReaction(descriptor.ngene, descriptor.is_or_integration)
    for nprot, is_activate in  \
        zip(descriptor.nprots, descriptor.is_activates):
      maker.addProtein(nprot, is_activate)
    maker.makeReaction()
    return maker
      

######################################################
class GeneNetwork(object):
  """
  Create a full model.
  Usage:
    network = GeneNetwork()
    network.update(<list-gene-descriptions>)
    network.update(<list-gene-descriptions>)
    network.do()
  Can then use network.model and network.parameters
  """

  def __init__(self, initial_network=INITIAL_NETWORK):
    """
    :param list-str gene_descriptors: gene descriptor strings
    """
    self._ngene = NGENE
    self._constants = []  # Constants in the model
    self.parameters = None  # lmfit.Parameters for model
    self.model = None  # Model string
    self._network = {}  # Key is gene; value is GeneDescriptor
    self._constants = []  # All constants in model
    self._initialize_constants = []  # Constants to initialize to 0
    self.updateDescriptions(initial_network, is_initialize=False)

  def update(self, strings, is_initialize=True):
    """
    Updates the entries for descriptors.
    :param list-str strings: list of gene descriptor strings
    :param bool is_intialize: constants should be initialized
    Updates
      self._constants
      self._initialize_constants
      self._network
    """
    for string in strings:
      new_desc = GeneDescriptor.parse(string)
      # Remove constants from the old descriptor
      if desc.ngene in self.network.keys():
        old_desc = self.network[desc.ngene]
        self._constants = list(set(self._constants).difference(old_desc))
        self._initialize_constants = list(set(self._constants).difference(old_desc))
      # Add constants for the new descriptor
      self.network[desc.ngene] = desc
      self._constants = self._constants.extend(desc.constants)
      if is_initialize:
        self._initialize_constants =  \
            self._initialize_constants.extend(desc.initialize_constants)
    # Verify that this is a complete network
    if len(self._network.keys()) != self._ngene:
      raise ValueError("Some key is not initialized: %s:"
          % str(self._network.keys()))

  def do(self):
    """
    Generates an antimony model for the gene network.
    Updates
      self.model
      self.parameters
    """
    # 1: Append the head of the file
    self.model = GeneNetwork._readFile(FILE_HEAD)
    # 2: Append gene reactions
    # 3: Append constant initializations
    # 4: Append the tail of the file
    self.model += "\n" + GeneNetwork._readFile(FILE_HEAD)
    # 5: Construct the lmfit.parameters for constants in the model
    self.parameters = mg.makeParameters(self.constants)

  def copy(self):
    """
    Copies the GeneReaction.
    :return GeneReaction:
    """
    return copy.deepcopy(self)

  @staticmethod
  def _readFile(path):
    with open(path) as fd:
      result = fd.readlines(path)
    return "\n".join(result)

  def __str__(self):
    return self._model
