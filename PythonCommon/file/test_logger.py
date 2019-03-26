'''Tests for logger.'''

import logger
import numpy as np
import os
import unittest


PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
FILEPATH =  os.path.join(PACKAGE_DIRECTORY, "test_logger.csv")
LOGGER = "TESTING"


class TestLogger(unittest.TestCase):

  def _construct(self):
    self.logger = logger.Logger(FILEPATH, LOGGER)

  def _getLines(self):
    fh = open(FILEPATH, "r")
    lines = fh.readlines()
    fh.close()
    return lines

  def testConstructor(self):
    self._construct()
    self.assertTrue(logger._HEADER in self._getLines()[0])

  def _testLog(self, construct=True, details=""):
    if construct:
      self._construct()
    name = "testing"
    self.logger.log("testing", details=details)

  def testLogOne(self):
    return
    self._testLog()
    full_name = "%s/%s" % (LOGGER, name)
    self.assertTrue(full_name in self._getLines()[1])

  def testLogTwo(self):
    self._testLog()
    self._testLog(construct=False, details='some details')
    self.assertEqual(len(self._getLines()), 3)

    


if __name__ == '__main__':
    unittest.main()
