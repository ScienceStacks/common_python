"""Tests for CategoricalCollection."""

from common_python.text.categorical_collection import CategoricalCollection
from common_python.testing import helpers

import os
import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = False

CC1 =  ['c', 'b', 'a']
CC2 =  ['c', 'a', 'b']


class TestCategoricalCollection(unittest.TestCase):

  def setUp(self):
    self.cate = CategoricalCollection(CC1)

  def testConstructor1(self):
    self.assertEqual(CC1, self.cate.cvalues)
    self.assertEqual(CC1, self.cate.sorted_cvalues)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
  unittest.main()
