"""Tests for Persister."""

from common_python.util import util
from common_python import constants as cn

import os
import unittest

IGNORE_TEST = False


class TestUtil(unittest.TestCase):

  def testChangeFileExtension(self):
    extension = "txt"
    filepath_without = "/tmp/aa"
    filepath_with = "%s.%s" % (filepath_without, extension)
    self.assertEqual(
        util.changeFileExtension(filepath_with, None),
        filepath_without)
    self.assertEqual(
        util.changeFileExtension(filepath_with, extension),
        filepath_with)
    

  def testGetValue(self):
    key = 'a'
    value = "value"
    dictionary = {key: value}
    self.assertTrue(value, util.getValue(dictionary, key, None))
    key = 'b'
    value = "value_b"
    self.assertTrue(value, util.getValue(dictionary, key, value))


if __name__ == '__main__':
    unittest.main()
