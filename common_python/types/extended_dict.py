'''Extends Dictionaries'''

import pandas as pd

from common_python.util.persister import Persister

class ExtendedDict(dict):

  def serialize(self, path):
    """
    :param str path: path to PCL file to write
    """
    persister = Persister(path)
    persister.set(self)

  @ classmethod
  def deserialize(cls, path):
    """
    :param str path: path to PCL file to read
    """
    persister = Persister(path)
    if not persister.isExist():
      raise ValueError(
          "Deserialization path does not exist.")
    return persister.get()

  def equals(self, other):
    """
    Checks if two dictionaries have the same
    keys and values.
    :param dict other:
    :return bool:
    Notes:
      key is an elemental type (str, int, float, bool)
      values are elemental type or list of element type
    """
    # Check keys
    diff = set(self.keys()).symmetric_difference(
        other.keys())
    if len(diff) != 0:
      return False
    for key in self.keys():
      if type(self[key]) != type(other[key]):
        return False
      this_value = self[key]
      other_value = other[key]
      if isinstance(this_value, list):
        if len(this_value) != len(other_value):
          return False
        result = all([t==o for t,o
            in zip(this_value, other_value)])
      else:
        result = this_value == other_value
      if not result:
        return False
    return True
    
    
