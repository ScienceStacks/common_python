'''Tests for utility routines.'''

import common_python.util.util as ut

import numpy as np
import pandas as pd
import sys
import unittest

IGNORE_TEST = False


class TestFunctions(unittest.TestCase):

  def testConvertType(self):
    if IGNORE_TEST:
      return
    self.assertEqual(ut.ConvertType('3'), 3)
    self.assertEqual(ut.ConvertType('3s'), '3s')
    self.assertTrue(abs(ut.ConvertType('3.1') - 3.1) < 0.001)

  def testConvertTypes(self):
    if IGNORE_TEST:
      return
    self.assertEqual(ut.ConvertTypes(['3', '3s', '3.1']),
                                    [ 3 , '3s',  3.1  ])

  def testRandomWord(self):
    if IGNORE_TEST:
      return
    WORDLEN = 7
    word = ut.randomWord(size=WORDLEN)
    self.assertTrue(isinstance(word, str))
    self.assertEqual(len(word), WORDLEN)

  def testGetValue(self):
    if IGNORE_TEST:
      return
    dictionary = {'a': 1}
    self.assertEqual(ut.getValue(dictionary, 'a', 0), 1)
    self.assertEqual(ut.getValue(dictionary, 'b', 0), 0)

  def testSetValue(self):
    if IGNORE_TEST:
      return
    dictionary = {'a': 1}
    new_dict = ut.setValue(dictionary, 'a', 2)
    self.assertEqual(new_dict['a'], 1)
    #
    new_dict = ut.setValue(dictionary, 'b', 2)
    self.assertEqual(new_dict['b'], 2)

  def testRandomWords(self):
    if IGNORE_TEST:
      return
    LEN = 10
    self.assertEqual(len(ut.randomWords(LEN)), LEN)

  def testGetFileExtension(self):
    if IGNORE_TEST:
      return
    extensions = ['x', 'xy', 'xyz']
    partial_filename = 'dummy'
    for ext in extensions:
      this_ext = ut.getFileExtension("%s.%s" %
          (partial_filename, ext))
      self.assertEqual(this_ext, ext)
    # Try with no extension
    this_ext = ut.getFileExtension("%s" % partial_filename)
    self.assertIsNone(this_ext)

  def testStripFileExtension(self):
    if IGNORE_TEST:
      return
    extensions = ['x', 'xy', 'xyz']
    partial_filename = '/u/dummy/xx'
    for ext in extensions:
      this_partial = ut.stripFileExtension("%s.%s" %
          (partial_filename, ext))
      self.assertEqual(this_partial, partial_filename)

  def testStripFileExtensionSingleFile(self):
    if IGNORE_TEST:
      return
    extensions = ['x', 'xy', 'xyz']
    partial_filename = 'dummy'
    for ext in extensions:
      this_partial = ut.stripFileExtension("%s.%s" % (partial_filename, ext))
      self.assertEqual(this_partial, partial_filename)

  def testChangeFileExtension(self):
    if IGNORE_TEST:
      return

    def createFilepath(ext):
      partial_filename = '/x/y/dummy'
      if ext is None:
        path = partial_filename
      else:
        path = "%s.%s" % (partial_filename, ext)
      return path

    extensions = ['x', 'xy', 'xyz', None]
    for from_ext in extensions:
      from_path = createFilepath(from_ext)
      for to_ext in extensions:
        to_path = createFilepath(to_ext)
        path = ut.changeFileExtension(from_path, to_ext)
        self.assertEqual(path, to_path)

  def testAddPath(self):
    if IGNORE_TEST:
      return
    repo_name = "common_python"
    def test(sub_dirs, checker_name):
      cur_path = list(sys.path)
      ut.addPath(repo_name, sub_dirs=sub_dirs)
      self.assertEqual(len(cur_path), len(sys.path) - 1)
      self.assertTrue(checker_name in sys.path[0])
    #
    test([], repo_name)
    test([repo_name, 'classifier'], 'classifier')

  def testInterpolateTime(self):
    if IGNORE_TEST:
      return
    MAX = 10
    SER = pd.Series(range(MAX), index=range(MAX))
    self.assertEqual(ut.interpolateTime(SER, 0.4), 0.4)
    self.assertEqual(ut.interpolateTime(SER, -1), 0)
    self.assertEqual(ut.interpolateTime(SER, MAX), MAX-1)

  def testMakeTimeInterpolationedMatrix(self):
    if IGNORE_TEST:
      return
    MAX = 5
    COLUMNS = ['a', 'b']
    FACTORS = [5, 10]
    df = pd.DataFrame({'time': range(MAX)})
    for n in range(len(COLUMNS)):
      df[COLUMNS[n]] = FACTORS[n]*df['time']
    df = df.set_index('time')
    #
    matrix = ut.makeTimeInterpolatedMatrix(df,
        num_interpolation=4)
    for idx in range(len(COLUMNS)):
      trues = [a[idx+1] == FACTORS[idx]*a[0]
          for a in matrix]
      self.assertTrue(all(trues))

  def testTrimeDF(self):
    if IGNORE_TEST:
      return
    data = np.repeat(2, 5)
    df = pd.DataFrame({'a': data, 'b': data})
    df_new = ut.trimDF(df)
    self.assertTrue(df.equals(df_new))
    #
    df_new = ut.trimDF(df, min_value=3)
    self.assertEqual(len(df_new), 0)
    # Test criteria
    criteria = lambda v: v < 3
    df_newer = ut.trimDF(df, criteria=criteria)
    self.assertTrue(df_newer.equals(df_new))
    # Test symmetric
    data = {'a': [1, -2, -3], 'b': [-2, -2, -3], 'c': [1, -3, -3]}
    df = pd.DataFrame(data, index=['a', 'b', 'c'])
    df_not_symmetric = ut.trimDF(df)
    df_symmetric = ut.trimDF(df, is_symmetric=True)
    self.assertGreater(len(df_symmetric), len(df_not_symmetric))

  def testTrimUnnamed(self):
    if IGNORE_TEST:
      return
    df = pd.DataFrame({'a': range(10), 'b': range(10)})
    df_new = ut.trimUnnamed(df)
    self.assertTrue(df.equals(df_new))
    df[ut.UNNAMED] = df['a']
    dff = ut.trimUnnamed(df)
    self.assertFalse(df_new.equals(df))
    self.assertTrue(dff.equals(df_new))

  def testIsSetEqual(self):
    if IGNORE_TEST:
      return
    SET1 = [1, 2, 3]
    SET2 = [1, 2]
    self.assertTrue(ut.isSetEqual(SET1, SET1))
    self.assertFalse(ut.isSetEqual(SET1, SET2))

  def testDecimalToBinary(self):
    if IGNORE_TEST:
      return
    NUM = 6 
    arr_bin = ut.decimalToBinary(NUM)
    arr_power = np.array([4, 2, 1])
    self.assertEqual(NUM,
        sum(arr_power*arr_bin))


if __name__ == '__main__':
  unittest.main()
