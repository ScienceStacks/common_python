"""
Constructs a gene network for the modeling game.

A gene is described by a descriptor string. There are
three cases.
1. 0 TF: g
2. 1 TF: gsp
3. 2 TF: gspisp

where: 
  g is the gene number
  i indicates the kind of integration:
      A for AND with non-competitive binding (impossible to have AND if competitive)
      O for OR with non-competitive binding
      P for OR with competitive binding
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
PATH_HEAD = os.path.join(CUR_DIR, FILE_HEAD)
PATH_TAIL = os.path.join(CUR_DIR, FILE_TAIL)
NUM_GENE = 8  # Number of genes
PLUS = "+"
# Initial network from the modeling game
INITIAL_NETWORK = [
    "1+0O+4", "2+4", "3+6", "4-2", 5, "6+7P-1", 7, "8-1"]
# Structure of gene string descriptor
POS_GENE = 0
POS_SIGN_1 = 1
POS_PROTEIN_1 = 2

# Initial arcs in reaction network
P0 = "INPUT"  # Protein 0 is the input
IDX_L = 0
IDX_DMRNA = 1

# Indexed by (is_competitive, is_or_integration)
CONJUNCTION_DICT = {
  (True, True): "P",
  (False, True): "O",
  (False, False): "A",
  }


################ Helper Functions ###################
def _setList(values):
  if values is None:
    return []
  else:
    return values

def _equals(list1, list2):
  diff = set(list1).symmetric_difference(list2)
  return len(diff) == 0

def _extendUnique(list1, list2):
  list1.extend(list2)
  return list(set(list1))


################ Classes ###################
class GeneDescriptor(object):
  # Describes the transcription factors (protein) for a gene
  # and how they affect the gene's activation

  def __init__(self, ngene, nprots=None, 
      is_activates=None, is_or_integration=True,
      is_competitive=False):
    """
    :param int ngene: number of the gene
    :param list-int nprots: list of protein TFs
    :param list-bool is_activates: list of how corresponding prot impacts gene
    :param bool is_or_integration: logic used to combine terms
    :param bool is_competitive binding:
    """
    self.ngene = int(ngene)
    self.nprots = [int(p) for p in _setList(nprots)]
    self.is_activates = _setList(is_activates)
    self.is_or_integration = is_or_integration
    self.is_competitive = is_competitive

  def equals(self, other):
    result = True
    result = result and (self.ngene == other.ngene)
    result = result and _equals(self.nprots, other.nprots)
    result = result and _equals(self.is_activates, other.is_activates)
    result = result and (
        self.is_or_integration == other.is_or_integration)
    return result

  def __repr__(self):
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
      conjunction = CONJUNCTION_DICT[
          (self.is_competitive, self.is_or_integration)]
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
    is_competitive = True
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
      conjunction_term = string[3]
      if not conjunction_term in CONJUNCTION_DICT.values():
        raise ValueError("Invalid integration term in descriptor: %s" % string)
      if (conjunction_term == "O") or (conjunction_term == "P"):
        is_or_integration = True
      else:
        is_or_integration = False
      if (conjunction_term == "P"):
        is_competitive = True
      else:
        is_competitive = False
      extractTF(string[4:6])
    #
    return GeneDescriptor(
        ngene,
        is_or_integration=is_or_integration,
        nprots=nprots,
        is_activates=is_activates,
        is_competitive=is_competitive)


######################################################
class GeneReaction(object):
  """Creates the reaction for gene production of mRNA."""
    
  def __init__(self, descriptor, is_or_integration=True):
    """
    :param int/GeneDescriptor ngene: Gene descriptor or int (if default)
    :param bool is_or_integration: logic used to combine terms
    """
    if not isinstance(descriptor, GeneDescriptor):
      ngene = descriptor
      descriptor = GeneDescriptor(ngene,
          is_or_integration=is_or_integration)
    # Public properties
    self.descriptor = descriptor
    self.constants = [
        self._makeVar("L"),          # IDX_L
        self._makeVar("d_mRNA"),     # IDX_DMRNA
        ]
    self.reaction = None
    # Private
    self._k_index = 0  # Index of the K constants
    self._H = self._makeVar("H")  # H constant
    self._Vm = self._makeVar("Vm")  # H constant
    self._mrna = self._makeVar("mRNA")
      
  def add(self, nprot, is_activate=True):
    """
    Adds a protein to the mRNA generation reaction for this gene.
    :param int nprot: numbers of the protein that is TF for the gene
    :param list-bool is_activation: whether the gene is activation (True)
    """
    nprots = list(set(self.descriptor.nprots).union([nprot]))
    is_activates = list(set(self.descriptor.is_activates).union(
        [is_activate]))
    self.descriptor = GeneDescriptor(
      self.descriptor.ngene,
      nprots=nprots,
      is_activates=is_activates,
      is_or_integration=self.descriptor.is_or_integration,
      )
      
  def _makeVar(self, name):
    return "%s%d" % (name, self.descriptor.ngene)

  def _makeKVar(self):
    self._k_index += 1
    var = "K%d_%d" % (self._k_index, self.descriptor.ngene)
    self.constants.append(var)
    return var

  @staticmethod
  def _makePVar(nprot):
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
      term = term + "*%s^%s" % (GeneReaction._makePVar(nprot),
          self._H)
    return term

  def _makeTFKinetics(self):
    """
    Creates the kinetics for the transcription factors
    """
    terms = [self._makeTerm([p]) for p in self.descriptor.nprots]
    if (len(self.descriptor.nprots) > 1) and (
        not self.descriptor.is_competitive):
      terms.append(self._makeTerm(self.descriptor.nprots))
    denominator = "1"
    for term in terms:
      denominator += " + %s" % term
    numerator = ""
    if self.descriptor.is_or_integration:
      for is_activate, term  in zip(
          self.descriptor.is_activates, terms[:-1]):
        if is_activate:
          numerator += "%s %s" %  (PLUS, term)
      if all(self.descriptor.is_activates) and (
          not self.descriptor.is_competitive):
        numerator += "%s %s" %  (PLUS, terms[-1])
      # Ensure that the numerator has at least on term
      if len(numerator) == 0:
        numerator = "1"  # Ensure has at least one term
    else:  # AND integration
      numerator = "1"
      if all(self.descriptor.is_activates):
        numerator = terms[-1]
      elif self.descriptor.is_activates[0]:
        numerator = terms[0]
      else:
        if len(self.descriptor.is_activates) > 1:
          if self.descriptor.is_activates[1]:
            numerator = terms[1]
    # Clean up the numerator by removing leading "+"
    splits = numerator.split(" ")
    if splits[0] == PLUS:
      numerator = " ".join(splits[1:])
    result = "%s * ( %s ) / ( %s )" % (
        self._Vm, numerator, denominator)
    return result
  
  def _makeKinetics(self):
    if len(self.descriptor.nprots) == 0:
      stg = self._makeBasicKinetics()
    else:
      stg = "%s + %s" % (self._makeBasicKinetics(),
          self._makeTFKinetics())
      self.constants.extend([
          self._Vm,
          self._H,
          ])
    return stg

  def generate(self):
    """
    Generates the reaction string.
    Updates:
      self.reaction
    """
    label = self._makeVar("J")
    self.reaction = "%s: => %s; %s" % (label, 
        self._mrna, self._makeKinetics())
    
  def __repr__(self):
    if self.reaction is None:
      self.generate()
    return self.reaction

  @classmethod
  def do(cls, descriptor):
    """
    Constructs the reaction and constants for the gene.
    :param str/GeneDescriptor descriptor: gene descriptor
    :return GeneReaction:
    """
    if not isinstance(descriptor, GeneDescriptor):
      descriptor = GeneDescriptor.parse(descriptor)
    reaction = GeneReaction(descriptor.ngene,
        descriptor.is_or_integration)
    for nprot, is_activate in  \
        zip(descriptor.nprots, descriptor.is_activates):
      reaction.add(nprot, is_activate)
    reaction.generate()
    return reaction
      

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
    self._ngene = NUM_GENE
    self._network = {}  # Key is gene; value is GeneDescriptor
    self._constants = []  # All constants in model
    self._uninitialize_constants = []  # Constants not initialized to 0
    self.update(initial_network, is_initialize=False)
    # Generated outputs
    self.parameters = None  # lmfit.Parameters for model
    self.model = None  # Model string

  def update(self, strings, is_initialize=True):
    """
    Updates the gene network.
    :param list-str strings: list of gene descriptor strings
    :param bool is_intialize: constants should be initialized
    Updates
      self._constants
      self._initialize_constants
      self._network
    """
    for string in strings:
      new_reaction = GeneReaction.do(string)
      # Remove constants from the old descriptor
      if new_reaction.descriptor.ngene in self._network.keys():
        old_reaction = self._network[new_reaction.descriptor.ngene]
        difference = set(self._constants).difference(
            old_reaction.constants)
        self._constants = list(difference)
      # Add the new reaction to the network
      self._network[new_reaction.descriptor.ngene] = new_reaction
      self._constants = _extendUnique(
          self._constants, new_reaction.constants)
      if not is_initialize:
        self._uninitialize_constants = _extendUnique(
            self._uninitialize_constants, new_reaction.constants)
    # Verify that this is a complete network
    if len(self._network.keys()) != self._ngene:
      raise RuntimeError("Some key is not initialized: %s:"
          % str(self._network.keys()))

  def generate(self):
    """
    Generates an antimony model for the gene network.
    Updates
      self.model
      self.parameters
    """
    # 1: Append the head of the file
    self.model = GeneNetwork._readFile(PATH_HEAD)
    # 2: Append gene reactions
    self.model += str(self)
    # 3: Append constant initializations
    comment = "\n\n// Initializations for new constants\n"
    self.model += comment
    constants = [v for v in self._constants
        if not v in self._uninitialize_constants]
    statements = "\n".join(["%s = 0;" % v for v in constants])
    self.model += statements
    # 4: Append the tail of the file
    self.model += "\n" + GeneNetwork._readFile(PATH_TAIL)
    # 5: Construct the lmfit.parameters for constants in the model
    self.parameters = mg.makeParameters(self._constants)

  def copy(self):
    """
    Copies the GeneReaction.
    :return GeneReaction:
    """
    return copy.deepcopy(self)

  @staticmethod
  def _readFile(path):
    with open(path, "r") as fd:
      result = fd.readlines()
    return "\n".join(result)

  def __repr__(self):
    return "\n".join([str(r) for r in self._network.values()])
