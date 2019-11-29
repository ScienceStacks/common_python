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
from common_python.testing import helpers

import lmfit
import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = True
DESC_STG = "7-7"
END_TIME = 300


###########################################################
class TestGeneAnalyzer(unittest.TestCase):

  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()

  def _init(self):
    self.analyzer = ga.GeneAnalyzer()
    self.analyzer._initializeODScope(DESC_STG, END_TIME)
    self.analyzer._initializeODPScope(
        self.analyzer.network.new_parameters)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self._init()
    self.assertTrue(
        isinstance(self.analyzer.parameters, lmfit.Parameters))
    self.assertTrue("P1" in self.analyzer.namespace.keys())
    self.assertTrue(helpers.isValidDataFrame(self.analyzer._df_mrna,
        self.analyzer._df_mrna.columns))

  def testMakePythonExpression(self):
    if IGNORE_TEST:
      return
    self.analyzer._initializeODScope(DESC_STG, END_TIME)
    result = ga.GeneAnalyzer._makePythonExpression(
        self.analyzer.reaction.mrna_kinetics)
    self.assertTrue(isinstance(eval(result, self.analyzer.namespace),
        float))

  def testCalcKinetics(self):
    if IGNORE_TEST:
      return
    y_arr = np.repeat(0, gn.NUM_GENE + 2)
    time = 0
    result = ga.GeneAnalyzer._calcKinetics(y_arr, time, self.analyzer)
    trues = [x >= 0 for x in result]
    self.assertTrue(all(trues))
    self.assertGreater(result[0], 0)

  def testCalcMrnaEstimates(self):
    if IGNORE_TEST:
      return
    self.assertTrue(self.analyzer.ser_est is None)
    self.analyzer._calcMrnaEstimates(
        self.analyzer.network.new_parameters)
    self.assertTrue(isinstance(self.analyzer.ser_est, pd.Series))
    self.assertEqual(len(self.analyzer.ser_est),
        END_TIME/ga.NUM_TO_TIME)

  def testDo(self):
    if IGNORE_TEST:
      return
    self._init()
    self.analyzer.do(DESC_STG, end_time=END_TIME)
    import pdb; pdb.set_trace()

  def testEulerOdeint(self):
    if IGNORE_TEST:
      return
    def square(_, time, __):
      return np.array(2*time)
    #
    self._init()
    MAX = 9
    result = self.analyzer.eulerOdeint(square, [0], range(MAX+1),
        num_iter=20)
    self.assertLess(np.abs(result[-1][0] - MAX*MAX), 1)

  def testProteinInitializations(self):
    if IGNORE_TEST:
      return
    df_mrna, compileds = ga.GeneAnalyzer.proteinInitializations(
        cn.MRNA_PATH)
    self.assertTrue("code" in str(compileds[0].__class__))
    self.assertTrue(helpers.isValidDataFrame(df_mrna,
        df_mrna.columns))

  def testProteinDydt(self):
    if IGNORE_TEST:
      return
    MAX = 10
    times = [10.0*n for n in range(MAX)]
    df_mrna, compileds = ga.GeneAnalyzer.proteinInitializations(
        cn.MRNA_PATH)
    y0_arr = np.repeat(0, gn.NUM_GENE + 1)
    y_arr = np.array(y0_arr)
    y_arrs = []
    for time in times:
      y_arr = ga.GeneAnalyzer._proteinDydt(y_arr, time,
          df_mrna, compileds)
      y_arrs.append(y_arr)
    trues = [np.isclose(v, 0) for v in y_arrs[0]]
    self.assertTrue(all(trues))
    trues = [v > 0.0 for v in y_arrs[-1]]
    self.assertTrue(all(trues))

  def testMakeProtein(self):
    if IGNORE_TEST:
      return
    df = ga.GeneAnalyzer.makeProteinDF(end_time=30)
    columns = [gn.GeneReaction.makeProtein(n)
        for n in range(1, gn.NUM_GENE+1)]
    columns.insert(0, cn.TIME)
    self.assertTrue(helpers.isValidDataFrame(df, columns))
  

if __name__ == '__main__':
  unittest.main()
