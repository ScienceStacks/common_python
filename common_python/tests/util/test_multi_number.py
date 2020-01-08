from common_python.util.multi_number import MultiNumber

import numpy as np
import pandas as pd
import unittest

IGNORE_TEST = False
BASES = [1, 2, 3]


class TestMultiNumber(unittest.TestCase):

  def setUp(self):
    self.multi = MultiNumber(BASES)

  def testRepr(self):
    if IGNORE_TEST:
      return
    stg = str(self.multi)
    self.assertEqual(stg.count("0"), len(BASES))

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertTrue(all(
        [x == y for x, y in zip(self.multi.bases, BASES)]))
    self.assertTrue(all( [x == 0 for x in self.multi.digits]))

  def testAddOne(self):
    if IGNORE_TEST:
      return
    self.multi._addOne(0)
    self.assertEqual(str(self.multi), "0, 1, 0")
    self.multi._addOne(0)
    self.assertEqual(str(self.multi), "0, 0, 1")

  def testNext(self):
    if IGNORE_TEST:
      return
    result = [x for x in self.multi]
    self.assertEqual(len(result), np.prod(BASES))
  
    
    


if __name__ == '__main__':
  unittest.main()
