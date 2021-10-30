'''Utility routines.''' 

import common_python.constants as cn

import os
import random
import sys
import pandas as pd
import matplotlib.colors as mcolors
import numpy as np
import string

LETTERS = string.ascii_lowercase
UNNAMED = "Unnamed: 0"
EXTRA_COLORS = np.array(list(mcolors.CSS4_COLORS.keys()))
new_order = np.random.permutation(list(range(
    len(EXTRA_COLORS))))
EXTRA_COLORS = EXTRA_COLORS[new_order]


def ConvertType(v):
  # Converts to int, float, str as required
  # Input: v - string representation
  # Output: r - new representation
  try:
    r = int(v)
  except:
    try:
      r = float(v)
    except:
      r = v  # Leave as string
  return r

def ConvertTypes(values):
  # Converts a list strings to a list of their types
  # Input: values - list
  # Output: results
  results = []
  for v in values:
    results.append(ConvertType(v))
  return results

def randomWords(count, size=5):
  # Generates a sequence of random words of the same size
  # Input: count - number of random words generated
  #        size - size of each word
  # Output: result - list of random words
  return [randomWord(size=size) for n in range(count)]

def randomWord(size=5):
  # Generates a random word
  # Input: size - size of each word
  # Output: word
  word = ''
  for n in range(size):
    word += random.choice(LETTERS)
  return word

# TODO: Add tests
def stringToClass(cls_str):
  """
  Converts the string representation of a class to a class object.
  :param str cls_str: string representation of a class
  :return type class:
  """
  import_stg1 = cls_str.split(" ")[1]
  import_stg2 = import_stg1.replace("'", "")
  import_stg3 = import_stg2.replace(">", "")
  import_parse = import_stg3.split(".")
  cls = import_parse[-1]
  import_path = '.'.join(import_parse[:-1])
  import_statement = "from %s import %s" % (import_path, cls)
  exec(import_statement)
  this_class = None
  assign_statement = "this_class = %s" % cls
  exec(assign_statement)
  return this_class

def getFileExtension(filepath):
  """
  :param str filepath:
  :return str: extension excluding the "."
  """
  extension = os.path.split(filepath)[-1]
  split_filename = extension.split('.')
  if len(split_filename) == 1:
    ext = None
  else:
    ext = split_filename[-1]
  return ext

def stripFileExtension(filepath):
  """
  :param str filepath:
  :return str:
 filepath without the extension
  """
  split_filepath = list(os.path.split(filepath))
  filename = split_filepath[-1]
  split_filename = filename.split('.')
  stripped_filename = split_filename[0]
  split_filepath[-1] = stripped_filename
  fullpath = ""
  for ele in split_filepath:
    fullpath = os.path.join(fullpath, ele)
  return fullpath

def changeFileExtension(filepath, extension):
  """
  :param str filepath:
  :param str extension: without "."
  :return str: filepath without the extension
  """
  stripped_filepath = stripFileExtension(filepath)
  if extension is None:
    return "%s" % stripped_filepath
  else:
    return "%s.%s" % (stripped_filepath, extension)

def getValue(dictionary, key, value):
  """
  Returns the value for the key in the dictionary or the default.
  :param dict dictionary:
  :param object key:
  :param object value:
  :return object:
  """
  if not key in dictionary.keys():
    return value
  else:
    return dictionary[key]

def setValue(dictionary, key, default_value):
  """
  Returns an updated dictionary set to the default value if
  none is present
  :param dict dictionary:
  :param object key:
  :param object default_value:
  :return dictionary:
  """
  value = getValue(dictionary, key, default_value)
  new_dict = dict(dictionary)
  new_dict[key] = value
  return new_dict

def setList(value):
  """
  Sets a list to empty if None.
  """
  if value is None:
    return []
  else:
    return value

def addPath(repo_name, sub_dirs=None):
  """
  Adds a path relative to the repository root.
  :param str repo_name:
  :param list-str sub_dirs:
  """
  if sub_dirs is None:
    sub_dirs = []
  path = os.path.dirname(os.path.abspath(__file__))
  done = False
  found_first_folder = False
  while not done:
    new_path, cur_folder  = os.path.split(path)
    if len(path) == 0:
      raise ValueError("Repo %s not found." % repo_name)
    if cur_folder == repo_name:
      if found_first_folder:
        root_folder = path
        done = True
        break
      else:
        found_first_folder = True
    path = new_path
  if not done:
    raise ValueError("Repository root of %s not found" % repo_name)
  for folder in sub_dirs:
    path = os.path.join(path, folder)
  sys.path.insert(0, path)

def interpolateTime(ser, time):
  """
  Interpolates a values between two times.
  :param pd.Series ser: index is time
  :param float time:
  :return float:
  """
  def findTime(a_list, func):
    if len(a_list) == 0:
      return np.nan
    else:
      return func(a_list)
  def findValue(time):
    if np.isnan(time):
      return np.nan
    else:
      return ser[time]
  #
  time_lb = findTime([t for t in ser.index if t <= time], max)
  time_ub = findTime([t for t in ser.index if t >= time], min)
  value_lb = findValue(time_lb)
  value_ub = findValue(time_ub)
  if np.isnan(value_lb):
    return value_ub
  if np.isnan(value_ub):
    return value_lb
  if time_ub == time_lb:
    return value_ub
  frac = (time - time_lb)/(time_ub - time_lb)
  return (1 - frac)*value_lb + frac*value_ub

def makeTimeInterpolatedMatrix(df, num_interpolation=10):
  """
  Does linear interpolations of values based on time.
  :param pd.DataFrame df: index is time
  :param int num_interpolation: number of interpolations between time
  :return np.array: first column is time
  Assumes that index is sorted ascending
  """
  times = df.index.tolist()
  time_last = times[0]
  matrix = []
  # For each pair of times
  for time in times[1:]:
    time_incr = (time - time_last)/num_interpolation
    arr_last = np.array(df.loc[time_last, :])
    arr_cur = np.array(df.loc[time, :])
    arr_incr = (arr_cur - arr_last)/num_interpolation
    # For each interpolation
    for idx in range(num_interpolation):
      arr = arr_last + idx*arr_incr
      arr = np.insert(arr, 0, time_last + idx*time_incr)
      matrix.append(arr)
    time_last = time
  return np.array(matrix)

def trimDF(df, min_value=0, is_symmetric=False,
                     criteria=None):
  """
  Remove columns and rows in which values satisfy a predicate.
  The default predicate is that values are less than min_value.
  :param pd.DataFrame df:
  :param bool is symmetric: only prune if same named row and
      column satisfy the predicate
  :param Function criteria: trimming criteria for a value
  :return pd.DataFrame:
  """
  if criteria is None:
      criteria = lambda v: v <= min_value
  delete_columns = []
  for col in df.columns:
    is_prune = all([criteria(v) for v in df[col]])
    if is_prune:
      delete_columns.append(col)
  delete_idxs = []
  for idx in df.index:
    is_prune = all([criteria(v) for v in df.loc[idx, :]])
    if is_prune:
      delete_idxs.append(idx)
  if is_symmetric:
    deletes = list(set(
      delete_columns).intersection(delete_idxs))
    delete_columns = deletes
    delete_idxs = deletes
  df_prune = df.copy()
  df_prune = df_prune.drop(delete_columns, axis=1)
  df_prune = df_prune.drop(delete_idxs, axis=0)
  if len(df_prune.columns) == 0:
    df_prune = pd.DataFrame()
  return df_prune

def trimUnnamed(df):
  """
  Removes "Unnamed" column if it exists.

  Parameters
  ----------
  df : pd.DataFrame

  Returns
  -------
  pd.DataFrame
  """
  df_new = df.copy()
  if UNNAMED in df_new.columns:
    del df_new[UNNAMED]
  return df_new

def isSetEqual(set1, set2):
  diff = set(set(set1)).symmetric_difference(set2)
  return len(diff) == 0

def decimalToBinary(dec):
  binary_stg = bin(dec)[2:]
  length = len(binary_stg)
  return np.array([int(binary_stg[i])
      for i in range(length)])

def makeTrinary(v):
  if v < 0:
    return -1
  elif v > 0:
    return 1
  else:
    return 0

def makeBinaryClass(v):
  if v < 0:
    return cn.NCLASS
  elif v > 0:
    return cn.PCLASS
  else:
    return np.nan

def convertSLToNumzero(sl, min_sl=1e-3):
  """
  Converts a (neg or pos) significance level to
  a count of significant zeroes.

  Parameters
  ----------
  sl: float

  Returns
  -------
  float
  """
  if np.isnan(sl):
    return 0
  if sl < 0:
    sl = min(sl, -min_sl)
    num_zero = np.log10(-sl)
  elif sl > 0:
    sl = max(sl, min_sl)
    num_zero = -np.log10(sl)
  else:
    raise RuntimeError("Cannot have significance level of 0.")
  return num_zero

def deserializePandas(path):
  """
  Retrives a pandas DataFrame or Series. There should be a column
  labelled "index". If there is only one non-index column, then
  a Series is returned.

  Parameters
  ----------
  path: str
    Path to CSV file
  
  Returns
  -------
  pd.Series/pd.DataFrame
  """
  UNNAMED = "Unnamed: 0"
  if path is None:
    return None
  df = pd.read_csv(path)
  if UNNAMED in df.columns:
    del df[UNNAMED]
  if not cn.INDEX in df.columns:
    raise ValueError("CSV file does not have a column labelled `index`: %s"
        % path)
  df = df.set_index("index")
  if len(df.columns) == 1:
    columns = list(df.columns)
    return df[columns[0]]
  else:
    return df

def serializePandas(pd_object, path):
  """
  Serializes a pandas DataFrame or Series.
  Creates a column for the index labelled "index".

  Parameters
  ----------
  pd_object: pd.Series, pd.DataFrame
  path: str
    Path to CSV file
  """
  if path is None:
    return
  #
  df = pd.DataFrame(pd_object.copy())
  df[cn.INDEX] = list(df.index)
  df.to_csv(path)

def findRepositoryRoot(repository_name):
  """
  Finds the root of the named repository.

  Parameters
  ----------
  repository_name: str
  
  Returns
  -------
  str: path to repository root
  """
  cur_path = os.path.abspath(".")
  done = False
  while not done:
    parent_path, cur_dir = os.path.split(cur_path)
    if cur_dir == repository_name:
      if not repository_name in parent_path:
        return cur_path
    if (len(parent_path) == 0) or (len(cur_dir) == 0):
      raise ValueError("Repository root not found: %s" % repository_name)
    cur_path = parent_path

def makeSymmetricDF(df):
  """
  Makes the dataframe symmetric if uncalculated values are np.nan.

  Parameters
  ----------
  df: pd.DataFrame
    index is same a columns
  
  Returns
  -------
  pd.DataFrame
  """
  df_result = df.fillna(0)
  df_result += df_result.T
  features = list(df_result.columns)
  df_result.loc[features, features] *= 0.5  # count diagonal once
  return df_result

def copyProperties(obj1, obj2):
  """
  Copies the properties __dict__ of obj1 to obj2.

  Parameters
  ----------
  obj1: object
  obj2: object
  """
  for name, value in obj1.__dict__.items():
    obj2.__dict__[name] = value

def isEqual(obj1, obj2):
  """
  Tests if two objects have the same instance dictionaries
  and values.
 
  Parameters
  ----------
  obj1: object
  obj2: object
 
  Returns
  -------
  bool
  """
  result = True
  result_info = None
  simple_types = [int, str, bool, list, dict]
  if (obj1 is None) and (obj2 is None):
    result_info = 0
    result = True
  elif obj1.__class__ in simple_types:
    result_info = 1
    result = obj1 == obj2
  elif obj1.__class__ == float:
    try:
      result_info = 2
      result = np.isclose(obj1, obj2)
    except Exception:
      result_info = 3
      result = False
  #
  elif "equals" in dir(obj1):
    result_info = 4
    result = obj1.equals(obj2)
  #
  else:
    for key in set(obj1.__dict__.keys()).union(obj2.__dict__.keys()):
      try:
        if not isEqual(obj1.__getattribute__(key),
            obj2.__getattribute__(key)):
          result_info = key
          result = False
      except Exception:
        result_info = 5
        result = False
  #
  return result

def getColors(count, excludes=None, includes=None):
  """
  Returns a list of unique colors
  Parameters
  ----------
  count: int - Number of colors requested
  excludes: list-str - colors to exclude
  includes: list-str - substrings of names that should be present
  
  Returns
  -------
  """
  def checkIncludes(color):
    if len(includes) == 0:
      return True
    result = any([c in color for c in includes])
    return result
  #
  def checkExcludes(color):
    result = all([c not in color for c in excludes])
    return result
  #
  if excludes is None:
    excludes = []
  if includes is None:
    includes = []
  colors = [c for c in list(mcolors.TABLEAU_COLORS) if checkExcludes(c)]
  #
  if count <= len(colors):
    return colors[:count]
  # Add more colors at random
  for idx in range(len(EXTRA_COLORS)):
    if len(colors) >= count:
      break
    new_color = EXTRA_COLORS[idx]
    if (checkIncludes(new_color) and
        checkExcludes(new_color) and (new_color not in colors)):
      colors.append(new_color)
    #
  return colors
