"""Tests for Term Analyzer."""

from common_python.text import util_text
from common_python.types.extended_list import ExtendedList
from common_python.testing import helpers

import os
import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = False

DF = pd.DataFrame({
    util_text.GROUP: ['a', 'b', 'b', 'c', 'a'],
    util_text.TERM: ['x', 'x', 'y', 'x', 'x'],
    })
DF = DF.set_index(util_text.GROUP)


class TestFunctions(unittest.TestCase):

  def testMakeTermMatrix(self):
    df = util_text.makeTermMatrix(DF[util_text.TERM])
    columns = ExtendedList(DF.columns.tolist())
    columns.unique()
    
    


if __name__ == '__main__':
  unittest.main()
