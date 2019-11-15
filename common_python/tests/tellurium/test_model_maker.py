"""Tests for classifier_ensemble.ClassifierEnsemble."""

from common_python.util import util
util.addPath("common_python", 
    sub_dirs=["common_python", "tellurium"])

from common_python.tellurium import model_maker

import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = False
NGENE = 1
NPROTS = [2, 3]
IS_ACTIVATES = [True, False]
NCONST_0_TF = 2  # L, d_mRNA
NCONST_1_TF = NCONST_0_TF + 3  # Vm, K1, H
NCONST_2_TF = NCONST_1_TF + 2  # K2, K3


class TestGeneMaker(unittest.TestCase):

  def setUp(self):
    self.maker = model_maker.GeneMaker(NGENE)

  def testConstructor(self):
    self.assertEqual(len(self.maker._nprots), 0)

  def _addProteins(self, maker, nprots, is_activates):
    is_activates = [True, False]
    [maker.addProtein(n, is_activate=b)
        for b, n in zip(is_activates, NPROTS)]

  def testAddProtein(self):
    self._addProteins(self.maker, NPROTS, IS_ACTIVATES)
    self.assertFalse(all(self.maker._is_activates))
    self.assertEqual(len(self.maker._is_activates),
      len(NPROTS))
    self.assertEqual(len(self.maker._nprots),
      len(NPROTS))
   
  def testMakePVar(self):
    nprot = 24
    self.assertEqual("P%d" % nprot, self.maker._makePVar(nprot))

  def testMakeBasicKinetics(self):
    stg = self.maker._makeBasicKinetics()
    self.assertTrue("L" in stg)
    self.assertTrue("d_mRNA" in stg)
    self.assertTrue("*mRNA" in stg)

  def testMakeTerm(self):
    stg = self.maker._makeTerm(NPROTS)
    self.assertTrue("K1" in stg)
    self.assertEqual(stg.count("P"), 2)
    stg = self.maker._makeTerm(NPROTS)
    self.assertTrue("K2" in stg)

  def testMakeTFKinetics(self):
    def test(is_or_integration):
      maker = model_maker.GeneMaker(NGENE,
          is_or_integration=is_or_integration)
      self._addProteins(maker, NPROTS, IS_ACTIVATES)
      stg = maker._makeTFKinetics()
      for n in range(1, 4):
        substg = "K%d" % n
        if not is_or_integration:
          break
        self.assertTrue(substg in stg)
      self.assertTrue("/" in stg)
    #
    test(True)
    test(False)

  def testMakeReaction(self):
    self._addProteins(self.maker, NPROTS, IS_ACTIVATES)
    self.maker.makeReaction()
    self.assertEqual(len(self.maker.constants), NCONST_2_TF)
    self.assertTrue("=>" in self.maker.reaction)
    for constant in self.maker.constants:
      self.assertTrue(constant in self.maker.reaction)

  def testMakeGeneDescriptor(self):
    desc = model_maker.GeneMaker._makeGeneDescriptor("2A+3")
    self.assertEqual(desc.ngene, 2)
    self.assertEqual(desc.is_or_integration, False)
    self.assertEqual(desc.is_activates[0], True)
    self.assertEqual(desc.nprots[0], 3)
    desc = model_maker.GeneMaker._makeGeneDescriptor("2O-3")
    desc = model_maker.GeneMaker._makeGeneDescriptor("2O-3,+4")
    self.assertEqual(len(desc.is_activates), 2)
 
  def testDo(self):
    # Descriptor for Gene 1
    desc_stg = "1O+0,+4"
    spec = model_maker.GeneMaker.do(desc_stg)
    self.assertEqual(len(spec.constants),  NCONST_2_TF)
    desc_stg = "4O-2"
    spec = model_maker.GeneMaker.do(desc_stg)
    self.assertEqual(len(spec.constants),  NCONST_1_TF)

  def testStr(self):
    # Smoke test
    self._addProteins(self.maker, NPROTS, IS_ACTIVATES)
    _ = str(self.maker)
    
    


if __name__ == '__main__':
  unittest.main()
