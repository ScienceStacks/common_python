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

  def compareOverlap(self, other, topN=None):
    """
    Calculates the overlap between two OrdinalCollections.
    Computes the ratio of the size of intersection to the size
    of the union.
    """
    if topN is None:
      topN1 = len(self.ordinals)
      topN2 = len(other.ordinals)
    else:
      topN1 = topN
      topN2 = topN
    set1 = set(self.ordinals[0:topN1])
    set2 = set(other.ordinals[0:topN2])
    num_intersection = len(set1.intersection(set2))
    num_union = len(set1.union(set2))
    return (1.0*num_intersection)/num_union
    
  def compareOrder(self, other, top=None):
    """
    Calculates the similarities in ordering of two
    OrdinalCollection.
    """
    pass
