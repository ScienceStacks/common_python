"""Tests for OrdinalCollection."""

from common_python.text.ordinal_collection import OrdinalCollection
from common_python.testing import helpers

import os
import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = False

OC1 =  ['c', 'b', 'a']
OC2 =  ['d', 'c', 'a', 'b']


class TestOrdinalCollection(unittest.TestCase):

  def setUp(self):
    self.collection = OrdinalCollection(OC1)

  def testConstructor1(self):
    self.assertEqual(OC1, self.collection.ordinals)

  def testMakeWithOrderings(self):
    orderings = [ [2, 3, 1], [30, 20, 10 ]]
    cc1 = list(OC1)
    collection = OrdinalCollection.makeWithOrderings(OC1, orderings)
    cc1.sort()
    self.assertEqual(cc1, collection.ordinals)
    #
    orderings = [ [-30, -20, -10 ]]
    collection = OrdinalCollection.makeWithOrderings(OC1, orderings,
        is_abs=False)
    self.assertEqual(OC1, collection.ordinals)
    #
    orderings = [ [-30, -20, -10 ]]
    collection = OrdinalCollection.makeWithOrderings(OC1, orderings,
        is_abs=True)
    cc1 = list(OC1)
    cc1.sort()
    self.assertEqual(cc1, collection.ordinals)

  def testCompareOverlap(self):
    other = OrdinalCollection(OC2)
    #
    result = self.collection.compareOverlap([other], topN=3)
    self.assertEqual(result, 1.0)
    #
    result = self.collection.compareOverlap([other])
    expected = (1.0*len(OC1))/len(OC2)
    self.assertEqual(result, expected)

  def testMakeOrderMatrix(self):
    return
    df = self.collection.makeOrderMatrix()
    import pdb; pdb.set_trace()



if __name__ == '__main__':
  unittest.main()
