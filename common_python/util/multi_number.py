"""Implementation of a number where each digit has its own base."""

import pandas as pd
import numpy as np

ONE = 1


class MultiNumber(object):

  def __init__(self, bases):
    """
    :param list-int bases:
    """
    self.bases = bases
    self.num_digits = len(bases)
    self._first = True
    # Value of the number
    self.digits = [0 for _ in range(self.num_digits)]

  def __str__(self):
    return ", ".join(["%d" % x for x in self.digits])

  def _addOne(self, pos):
    """
    Adds one to the number at the specified position, handling
    carries.
    :param int pos:
    :raises StopIteration:
    """
    if pos >= len(self.bases):
      raise StopIteration
    if self.digits[pos] < self.bases[pos] - ONE:
      self.digits[pos] += ONE
    else:
      self.digits[pos] = 0
      self._addOne(pos + 1)

  def __iter__(self):
    return self

  def __next__(self):
    if self._first:
      self._first = False
      return list(self.digits)
    self._addOne(0)
    return list(self.digits)

