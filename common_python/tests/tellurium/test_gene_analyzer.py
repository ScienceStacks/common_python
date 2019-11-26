from common_python.util import util
util.addPath("common_python", 
    sub_dirs=["common_python", "tellurium"])

from common_python.tellurium import model_fitting as mf
from common_python.tellurium import modeling_game as mg
from common_python.tellurium import gene_network as gn
from common_python.tellurium import gene_analyzer as ga
from common_python.tellurium.gene_network import  \
    GeneDescriptor, GeneReaction, GeneNetwork

import lmfit
import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = False
DESC_STG = "7+1A-7"


###########################################################
class TestGeneAnalyzer(unittest.TestCase):

  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()

  def _init(self):
    if IGNORE_TEST:
      return
    self.df_mrna = mf.cleanColumns(pd.read_csv("wild.csv"))
    self.analyzer = ga.GeneAnalyzer(self.df_mrna, DESC_STG)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertTrue(isinstance(self.analyzer.parameters,
        lmfit.Parameters))
    self.assertGreater(
        len(self.analyzer.parameters.valuesdict().keys()), 2)
    self.assertTrue("P1" in self.analyzer.namespace.keys())

  def testTransformKinetics(self):
    result = ga.GeneAnalyzer._transformKinetics(
        self.analyzer._gene_reaction.mrna_kinetics)
    self.assertTrue(isinstance(eval(result, self.analyzer.namespace),
        float))


if __name__ == '__main__':
  unittest.main()
