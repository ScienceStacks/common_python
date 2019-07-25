'''Analyzes collections of ordinal values.'''
"""
An ordinal value (ovalue) is an object with a specified order
relative to other objects in its collection. The default
order is based on position in the collection.

Ordinal collections can be compared in the following ways:
1. overlap. The fraction of the values of both that are shared.
            |A interesection B| / |A union B|
2. ordering. The fraction of pairwise comparisons of weights
             of categorical values that are the same in both
             categorical collections.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class OrdinalCollection(object):

  def __init__(self, ordinals):
    """
    :param list-object ordinal_values: ordinals
        values are ordered from smallest to largest
        in ordinal value
    """
    self.ordinals = list(ordinals)

  @classmethod
  def makeWithOrderings(cls, ordinals, orderings, is_abs=True):
    """
    Creates an OrdinalCollection using the list of orderings.
    :param list-list-float orderings: the inner list has
        one value for each ordinal
    :param bool is_abs: use absolute value
    :return OrdinalCollection:
    """
    if is_abs:
      func = np.abs
    else:
      func = lambda v: v  # identity function
    # Terms in order sorted by weight
    adjusted_orderings = [[func(x) for x in xv] for xv in orderings]
    keys = [max(xv) for xv in zip(*adjusted_orderings)]
    sorted_ordinals = [o for _, o in sorted(zip(keys, ordinals))]
    return cls(sorted_ordinals)

  @staticmethod
  def _calcTopN(ordinals, topN):
    if topN is None:
      topN = len(ordinals)
    return ordinals[-topN:]

  def compareOverlap(self, others, topN=None):
    """
    Calculates the overlap between two OrdinalCollections.
    Computes the ratio of the size of intersection to the size
    of the union.
    :param list-OrdinalCollection other:
    """
    cls = self.__class__
    #
    ordinal_sets = [set(cls._calcTopN(self.ordinals, topN))]
    for other in others:
      ordinal_sets.append(set(cls._calcTopN(other.ordinals, topN)))
    ordinal_union = set([])
    for other in ordinal_sets:
      ordinal_union = ordinal_union.union(other)
    ordinal_intersection = ordinal_union
    for other in ordinal_sets:
      ordinal_intersection = ordinal_intersection.intersection(other)
    num_intersection = len(ordinal_intersection)
    num_union = len(ordinal_union)
    return (1.0*num_intersection)/num_union

  def makeOrderMatrix(self):
    """
    Create a matrix where a 1 in cell ij means that ordinal
    i is less or equal to ordinal j.
    """
    length = len(self.ordinals)
    df = pd.DataFrame(np.repeat(0, length, length))
    df.columns = self.ordinals
    df.index = self.ordinals
    for idx, ordinal in enumerate(self.ordinals):
      df.loc[ordinal, range(idx, length)] = 1
    return df
    
  def compareOrder(self, other, top=None):
    """
    Calculates the similarities in ordering of two
    OrdinalCollection.
    """
    ordinals1 = cls._calcTopNSet(self.ordinals)
    ordinals2 = cls._calcTopNSet(other.ordinals)
