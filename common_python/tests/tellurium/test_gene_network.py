""" Tests for classifier_ensemble.ClassifierEnsemble."""

from common_python.util import util
util.addPath("common_python", 
    sub_dirs=["common_python", "tellurium"])

from common_python.tellurium import gene_network

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
# GeneDescriptor
ACTIVATE_1 = "+"
PROTEIN_1 = "3"
ACTIVATE_2 = "-"
PROTEIN_2 = "4"
INTEGRATE = "A"
GENE_0_TF = str(NGENE)
GENE_1_TF = "%d%s%s" % (NGENE, ACTIVATE_1, PROTEIN_1)
GENE_2_TF = "%d%s%s%s%s%s" % (
    NGENE, ACTIVATE_1, PROTEIN_1, INTEGRATE, ACTIVATE_2, PROTEIN_2)


###########################################################
class TestGeneDescriptor(unittest.TestCase):

  def setUp(self):
    self.descriptor = gene_network.GeneDescriptor.parse(GENE_2_TF)

  def testConstructor(self):
    descriptor = gene_network.GeneDescriptor(NGENE,
        nprots=[PROTEIN_1, PROTEIN_2],
        is_activates=[True, False],
        is_or_integration=False)
    self.assertTrue(self.descriptor.equals(descriptor))

  def testParse(self):
    desc = gene_network.GeneDescriptor.parse("2")
    self.assertEqual(desc.ngene, 2)
    desc = gene_network.GeneDescriptor.parse(2)
    self.assertEqual(desc.ngene, 2)
    desc = gene_network.GeneDescriptor.parse("2+3")
    self.assertEqual(desc.ngene, 2)
    self.assertEqual(desc.is_activates[0], True)
    self.assertEqual(desc.nprots[0], 3)
    desc = gene_network.GeneDescriptor.parse("2-3")
    self.assertEqual(desc.is_activates[0], False)
    self.assertEqual(desc.nprots[0], 3)
    desc = gene_network.GeneDescriptor.parse("2-3O+4")
    self.assertEqual(len(desc.is_activates), 2)
    self.assertEqual(desc.is_activates[0], False)
    self.assertEqual(desc.nprots[0], 3)
    self.assertEqual(desc.is_activates[1], True)
    self.assertEqual(desc.nprots[1], 4)
    

###########################################################
class TestGeneReaction(unittest.TestCase):

  def setUp(self):
    self.maker = gene_network.GeneReaction(NGENE)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(len(self.maker._nprots), 0)

  def _addProteins(self, maker, nprots, is_activates):
    if IGNORE_TEST:
      return
    is_activates = [True, False]
    [maker.addProtein(n, is_activate=b)
        for b, n in zip(is_activates, NPROTS)]

  def testAddProtein(self):
    if IGNORE_TEST:
      return
    self._addProteins(self.maker, NPROTS, IS_ACTIVATES)
    self.assertFalse(all(self.maker._is_activates))
    self.assertEqual(len(self.maker._is_activates),
      len(NPROTS))
    self.assertEqual(len(self.maker._nprots),
      len(NPROTS))
   
  def testMakePVar(self):
    if IGNORE_TEST:
      return
    nprot = 24
    self.assertEqual("P%d" % nprot, self.maker._makePVar(nprot))

  def testMakeBasicKinetics(self):
    if IGNORE_TEST:
      return
    stg = self.maker._makeBasicKinetics()
    self.assertTrue("L" in stg)
    self.assertTrue("d_mRNA" in stg)
    self.assertTrue("*mRNA" in stg)

  def testMakeTerm(self):
    if IGNORE_TEST:
      return
    stg = self.maker._makeTerm(NPROTS)
    self.assertTrue("K1" in stg)
    self.assertEqual(stg.count("P"), 2)
    stg = self.maker._makeTerm(NPROTS)
    self.assertTrue("K2" in stg)

  def testMakeTFKinetics(self):
    if IGNORE_TEST:
      return
    def test(is_or_integration):
      maker = gene_network.GeneReaction(NGENE,
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
    if IGNORE_TEST:
      return
    self._addProteins(self.maker, NPROTS, IS_ACTIVATES)
    self.maker.makeReaction()
    self.assertEqual(len(self.maker.constants), NCONST_2_TF)
    self.assertTrue("=>" in self.maker.reaction)
    for constant in self.maker.constants:
      self.assertTrue(constant in self.maker.reaction)
 
  def testDo(self):
    if IGNORE_TEST:
      return
    desc_stg = "8-1"
    spec = gene_network.GeneReaction.do(desc_stg)
    desc_stg = "1+0O+4"
    maker = gene_network.GeneReaction.do(desc_stg)
    self.assertEqual(len(maker.constants),  NCONST_2_TF)
    desc_stg = "4-2"
    maker = gene_network.GeneReaction.do(desc_stg)
    self.assertEqual(len(maker.constants),  NCONST_1_TF)

  def testStr(self):
    if IGNORE_TEST:
      return
    # Smoke test
    self._addProteins(self.maker, NPROTS, IS_ACTIVATES)
    _ = str(self.maker)
    

###########################################################
class TestGeneNetwork(unittest.TestCase):

  def setUp(self):
    self.network = gene_network.GeneNetwork()
    
    


if __name__ == '__main__':
  unittest.main()
