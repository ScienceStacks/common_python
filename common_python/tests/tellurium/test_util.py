from common_python.tellurium import util

import pandas as pd
import numpy as np
import unittest


class TestFunctions(unittest.TestCase):

  def testDfToSer(self):
    data = range(5)
    df = pd.DataFrame({'a': data, 'b': [2*d for d in data]})
    ser = util.dfToSer(df)
    assert(len(ser) == len(df.columns)*len(df))
  
  def testDfToSer(self):
    data = range(5)
    df = pd.DataFrame({'a': data, 'b': [2*d for d in data]})
    ser = util.dfToSer(df)
    assert(len(ser) == len(df.columns)*len(df))


if __name__ == '__main__':
  unittest.main()
