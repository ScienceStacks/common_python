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


class TestReactionMaker(unittest.TestCase):

  def setUp(self):
    self.maker = model_maker.ReactionMaker(NGENE)

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
    self.assertTrue("_1" in stg)
    self.assertEqual(stg.count("P"), 2)
    stg = self.maker._makeTerm(NPROTS)
    self.assertTrue("_2" in stg)

  def testMakeTFKinetics(self):
    def test(is_or_integration):
      maker = model_maker.ReactionMaker(NGENE,
          is_or_integration=is_or_integration)
      self._addProteins(maker, NPROTS, IS_ACTIVATES)
      stg = maker._makeTFKinetics()
      for n in range(1, 4):
        substg = "_%d" % n
        if not is_or_integration:
          break
        self.assertTrue(substg in stg)
      self.assertTrue("/" in stg)
    #
    test(True)
    test(False)

  def testMakeReaction(self):
    self._addProteins(self.maker, NPROTS, IS_ACTIVATES)
    stg = self.maker.makeReaction()
    self.assertEqual(len(self.maker._constants),
      len(NPROTS) + 1 + 4)
    self.assertTrue("=>" in stg)
    for constant in self.maker._constants:
      self.assertTrue(constant in stg)
  

if __name__ == '__main__':
  unittest.main()
