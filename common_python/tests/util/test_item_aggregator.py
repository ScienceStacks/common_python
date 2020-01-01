import pandas as pd
import unittest

import common_python.constants as cn
from common_python.util.item_aggregator import ItemAggregator

IGNORE_TEST = False
SIZE = 10
MULT = 5
ITEMS = [(n, MULT*n) for n in range(SIZE)]

class TestItemAggregator(unittest.TestCase):

  def setUp(self):
    self.aggregator = ItemAggregator(lambda v: v[0])

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertIsNone(self.aggregator._df)

  def testAppend(self):
    if IGNORE_TEST:
      return
    self.aggregator.append(ITEMS)
    self.assertEqual(len(self.aggregator.sers[0]), SIZE)
    self.aggregator.append(ITEMS)
    self.assertEqual(len(self.aggregator.sers), 2)

  def testDf(self):
    if IGNORE_TEST:
      return
    aggregator1 = ItemAggregator(lambda v: v[1])
    for agg in [self.aggregator, aggregator1]:
      agg.append(ITEMS)
      agg.append(ITEMS)
    df = MULT*self.aggregator.df
    self.assertTrue(aggregator1.df[cn.MEAN].equals(df[cn.MEAN]))
  


if __name__ == '__main__':
  unittest.main()
