"""Tests for Extended list"""

from common_python.types.extended_list import ExtendedList

import unittest

IGNORE_TEST = False
LIST = [0, 1, 2, 2, 3]


class TestPersister(unittest.TestCase):

  def setUp(self):
    self.extended_list = ExtendedList(LIST)

  def testRemoveAll(self):
    self.extended_list.removeAll(2)
    self.assertEqual(self.extended_list.count(2), 0)

  def testUnique(self):
    self.extended_list.unique()
    for ele in LIST:
      self.assertTrue(self.extended_list.count(ele) == 1)


if __name__ == '__main__':
  unittest.main()
