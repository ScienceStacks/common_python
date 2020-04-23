"""Tests for Extended dict"""

from common_python.types.extended_dict import ExtendedDict
import common_python.constants as cn


import copy
import os
import unittest

IGNORE_TEST = False
TEST_PATH = os.path.join(cn.TEST_DIR,
    "test_extended_dict.csv")
DICT = {'a': [0, 1], 'b': [0, 1, 2, 3], 'c': 4}


class TestExtendedDict(unittest.TestCase):

  def _remove(self):
    if os.path.isfile(TEST_PATH):
      os.remove(TEST_PATH)

  def setUp(self):
    self.extended_dict = ExtendedDict(DICT)
    self._remove()

  def tearDown(self):
    self._remove()

  def testSerialize(self):
    if IGNORE_TEST:
      return
    self.extended_dict.serialize(TEST_PATH)
    self.assertTrue(os.path.isfile(TEST_PATH))
 
  def testEquals(self):
    if IGNORE_TEST:
      return
    other = ExtendedDict(DICT)
    self.assertTrue(self.extended_dict.equals(other))
    #
    other_t = copy.deepcopy(other)
    other_t['a'].append(33)
    self.assertFalse(self.extended_dict.equals(other_t))
    #
    other_t = copy.deepcopy(other)
    other_t['a'][0] = 100
    self.assertFalse(self.extended_dict.equals(other_t))

  def testDeserialize(self):
    if IGNORE_TEST:
      return
    self.extended_dict.serialize(TEST_PATH)
    dct = ExtendedDict.deserialize(TEST_PATH)
    self.assertTrue(dct.equals(self.extended_dict))


if __name__ == '__main__':
  unittest.main()
