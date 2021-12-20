'''Persists an object as a Pickle file.'''

import pickle
import os


class Persister(object):
  """
  A light wrapper around pickle to persist an object.
  """

  def __init__(self, path):
    """
    :param str path: path for the pickle file persistence.
    """
    self.path = path

  def __repr__(self):
    return "Persister for %s" % self.path

  def isExist(self):
    return os.path.isfile(self.path)
    
  def set(self, an_object):
    with open(self.path, 'wb') as fd:
      pickle.dump(an_object, fd)

  def get(self):
    """
    :return object:
    Will throw FileNotFoundError if self.path does not exist
    """
    with open(self.path, 'rb') as fd:
      return pickle.load(fd)

  def remove(self):
    """
    Deletes the persistence file.
    """
    if self.isExist():
      os.remove(self.path)
