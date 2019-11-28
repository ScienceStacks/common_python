from common_python.util import util
util.addPath("common_python", 
    sub_dirs=["common_python", "tellurium"])

from common_python.tellurium import constants as cn
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
DESC_STG = "7"


###########################################################
class TestGeneAnalyzer(unittest.TestCase):

  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()

  def _init(self):
    if IGNORE_TEST:
      return
    self.df_mrna = mf.cleanColumns(pd.read_csv(cn.MRNA_PATH))
    self.analyzer = ga.GeneAnalyzer(self.df_mrna)
    self.analyzer._initializeODScope(DESC_STG)
    self.analyzer._initializeODPScope(
        self.analyzer.network.new_parameters)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertTrue(
        isinstance(self.analyzer.parameters, lmfit.Parameters))
    self.assertTrue("P1" in self.analyzer.namespace.keys())

  def testMakePythonExpression(self):
    self.analyzer._initializeODScope(DESC_STG)
    result = ga.GeneAnalyzer._makePythonExpression(
        self.analyzer.reaction.mrna_kinetics)
    self.assertTrue(isinstance(eval(result, self.analyzer.namespace),
        float))

  def testCalcKinetics(self):
    y_arr = np.repeat(0, gn.NUM_GENE + 2)
    time = 0
    result = ga.GeneAnalyzer._calcKinetics(y_arr, time, self.analyzer)
    trues = [x >= 0 for x in result]
    self.assertTrue(all(trues))
    self.assertGreater(result[0], 0)

  def testCalcMrnaEstimates(self):
    self.assertTrue(self.analyzer.ser_est is None)
    self.analyzer._calcMrnaEstimates(
        self.analyzer.network.new_parameters)
    self.assertTrue(isinstance(self.analyzer.ser_est, pd.Series))
    self.assertEqual(len(self.analyzer.ser_est),
        len(self.analyzer._df_mrna))

  def testDo(self):
    self.analyzer.do(DESC_STG)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
  unittest.main()
