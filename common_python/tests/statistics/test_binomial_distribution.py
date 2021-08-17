from common_python.statistics import binomial_distribution
import common_python.constants as cn
from common_python.testing import helpers

import numpy as np
import pandas as pd
import unittest


IGNORE_TEST = False
IS_PLOT = False
COUNT = 100
PROB = 0.5


class TestBinomialDistribution(unittest.TestCase):

  def setUp(self):
    self.binom = binomial_distribution.BinomialDistribution(COUNT,
       binomial_prob=PROB)
    
  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(COUNT, self.binom.max_count)

  def testPopulateSignificanceLevels(self):
    if IGNORE_TEST:
      return
    def test(mat):
      self.assertEqual(np.shape(self.binom.pos_sl_mat), (COUNT + 1, COUNT + 1))
    #
    test(self.binom.pos_sl_mat)
    test(self.binom.neg_sl_mat)

  def testGetSL1(self):
    if IGNORE_TEST:
      return
    num_sample = 11
    for num_event in range(num_sample//2):
      prob1 = self.binom.getSL(num_sample, num_event)
      prob2 = self.binom.getSL(num_sample, num_sample - num_event)
      self.assertTrue(np.isclose(np.abs(prob1), np.abs(prob2)))

  def testGetSL2(self):
    # Check case of probaility of pos event != prob neg event
    if IGNORE_TEST:
      return
    num_sample = 11
    prob = 0.75
    binom = binomial_distribution.BinomialDistribution(COUNT,
       binomial_prob=prob)
    for num_event in range(num_sample//2):
      prob = binom.getSL(num_sample, num_event)
      # Should be unlikely to get negative events and so their occurrence
      # is significant for a larger number of events
      self.assertLess(prob, 0)


  def testIsLowSL(self):
    if IGNORE_TEST:
      return
    num_sample = 11
    self.assertTrue(self.binom.isLowSL(num_sample, 0, 0.5))
    self.assertTrue(self.binom.isLowSL(num_sample, num_sample, 0.5))
    self.assertFalse(self.binom.isLowSL(num_sample, num_sample//2, 0.5))
    

if __name__ == '__main__':
  unittest.main()
