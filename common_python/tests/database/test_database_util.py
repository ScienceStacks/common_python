from common_python.database import database_util as util
import common_python.constants as cn


import os
import numpy as np
import pandas as pd
import sys
import unittest

IGNORE_TEST = False
TEST_DB_PTH = os.path.join(cn.TEST_DIR, "test_database_util.db")
TEST_CSV_PTH = os.path.join(cn.TEST_DIR, "test_database_util.csv")
SIZE = 4
COL_A = " a A "
COL_B = "b"
DF = pd.DataFrame({COL_A: range(SIZE)})
DF[COL_B] = 10*DF[COL_A]


class TestFunctions(unittest.TestCase):

  def _cleanUp(self):
    for path in [TEST_DB_PTH, TEST_CSV_PTH]:
      if os.path.isfile(path):
        os.remove(path)

  def setUp(self):
    self._cleanUp()

  def tearDown(self):
    self._cleanUp()

  def testCsvToTable(self):
    if IGNORE_TEST:
      return
    DF.to_csv(TEST_CSV_PTH, index=False)
    util.csvToTable(TEST_CSV_PTH, TEST_DB_PTH)
    self.assertTrue(os.path.isfile(TEST_DB_PTH))

if __name__ == '__main__':
  unittest.main()
