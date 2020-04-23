"""Tests for Extended dict"""

from common_python.types.extended_dict import ExtendedDict
import common_python.constants as cn


import copy
import os
import unittest

IGNORE_TEST = False
TEST_PATH_PCL = os.path.join(cn.TEST_DIR,
    "test_extended_dict.pcl")
TEST_PATH_CSV = os.path.join(cn.TEST_DIR,
    "test_extended_dict.csv")
DICT = {'a': [0, 1], 'b': [0, 1, 2, 3], 'c': 4}


class TestExtendedDict(unittest.TestCase):

  def _remove(self, path=TEST_PATH_PCL):
    if os.path.isfile(TEST_PATH_PCL):
      os.remove(TEST_PATH_PCL)

  def setUp(self):
    self.extended_dict = ExtendedDict(DICT)
    self._remove()

  def tearDown(self):
    self._remove()
    self._remove(path=TEST_PATH_CSV)

  def testSerialize(self):
    if IGNORE_TEST:
      return
    self.extended_dict.serialize(TEST_PATH_PCL)
    self.assertTrue(os.path.isfile(TEST_PATH_PCL))
 
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
    self.extended_dict.serialize(TEST_PATH_PCL)
    dct = ExtendedDict.deserialize(TEST_PATH_PCL)
    self.assertTrue(dct.equals(self.extended_dict))

  def testTo_csv(self):
    if IGNORE_TEST:
      return
    self._remove(path=TEST_PATH_CSV)
    self.extended_dict.toCsv(TEST_PATH_CSV)
    self.assertTrue(os.path.isfile(TEST_PATH_CSV))


if __name__ == '__main__':
  unittest.main()
