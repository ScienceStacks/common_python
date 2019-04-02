"""Tests for Term Analyzer."""

from common_python.text import term_analyzer
from common_python import constants as cn
from common_python.testing import helpers

import os
import pandas as pd
import unittest

IGNORE_TEST = False

SER = pd.Series(['a bb c', 'aa bb c', 'a c'])


class TestTermAnalyzer(unittest.TestCase):

  def setUp(self):
    self.analyzer = term_analyzer.TermAnalyzer()

  def testConstructor(self):
    trues = [x == y for x, y in 
        zip(self.analyzer._noise_terms, term_analyzer.NOISE_TERMS)]
    self.assertTrue(all(trues))

  def testMakeDF(self):
    self.analyzer.makeDF(SER)
    self.assertTrue(helpers.isValidDataFrame(self.analyzer.df_term,
        [cn.COUNT, cn.FRAC]))

if __name__ == '__main__':
  unittest.main()
