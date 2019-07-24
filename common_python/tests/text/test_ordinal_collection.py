"""Tests for OrdinalCollection."""

from common_python.text.ordinal_collection import OrdinalCollection
from common_python.testing import helpers

import os
import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = False

CC1 =  ['c', 'b', 'a']
CC2 =  ['c', 'a', 'b']


class TestOrdinalCollection(unittest.TestCase):

  def setUp(self):
    self.collection = OrdinalCollection(CC1)

  def testConstructor1(self):
    self.assertEqual(CC1, self.collection.ordinals)

  def testMakeWithOrderings(self):
    orderings = [ [2, 3, 1], [30, 20, 10 ]]
    cc1 = list(CC1)
    collection = OrdinalCollection.makeWithOrderings(CC1, orderings)
    cc1.sort()
    self.assertEqual(cc1, collection.ordinals)
    #
    orderings = [ [-30, -20, -10 ]]
    collection = OrdinalCollection.makeWithOrderings(CC1, orderings,
        is_abs=False)
    self.assertEqual(CC1, collection.ordinals)
    #
    orderings = [ [-30, -20, -10 ]]
    collection = OrdinalCollection.makeWithOrderings(CC1, orderings,
        is_abs=True)
    cc1 = list(CC1)
    cc1.sort()
    self.assertEqual(cc1, collection.ordinals)


if __name__ == '__main__':
  unittest.main()
