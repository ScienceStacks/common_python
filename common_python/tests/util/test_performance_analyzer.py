'''Tests for utility routines.'''

import common_python.util.performance_analyzer as pa

import unittest

IGNORE_TEST = False
IS_PLOT = False
NAME = "name"


class TestPerformanceAnalyzer(unittest.TestCase):

    def setUp(self):
        self.perf = pa.PerformanceAnalyzer()

    def testConstructor(self):
        if IGNORE_TEST:
          return
        self.assertTrue(isinstance(self.perf.elapsed_dct, dict))

    def testStart(self):
        if IGNORE_TEST:
          return
        self.perf.start(NAME)
        self.assertEqual(len(self.perf.start_dct), 1)

    def testEnd(self):
        if IGNORE_TEST:
          return
        self.perf.start(NAME)
        self.perf.end(NAME)
        self.assertEqual(len(self.perf.start_dct), 1)
        self.assertLess(self.perf.elapsed_dct[NAME], 1e4)


if __name__ == '__main__':
  unittest.main()
