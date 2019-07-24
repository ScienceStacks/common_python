'''Analyzes collections of categorical values.'''
"""
A categorical value (cvalue) is an object, 
and a categorical collection
is a collection of categorical values. Categorical values have
a weight that determines their ordering. The default weight
is the position of the categorical value in the collection.

Categorical collections can be compared in the following ways:
1. overlap. The fraction of the values of both that are shared.
            |A interesection B| / |A union B|
2. ordering. The fraction of pairwise comparisons of weights
             of categorical values that are the same in both
             categorical collections.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CategoricalCollection(object):

  def __init__(self, terms, raw_weights=None):
    """
    :param list-object terms: categorical items
    :param list-list-float raw_weights: multiple instances
        of weights for each term
    """
    self.cvalues = terms
    self.raw_weights = raw_weights
    # Choose the weight with the largest absolute value
    if self.raw_weights is None:
      self.weights = [-x for x in range(len(self.cvalues))]
    else:
      self.weights = self.__class__._calcWeights(self.raw_weights)
    # Terms in order sorted by weight
    self.sorted_cvalues = [x for _, x in 
        sorted(zip(self.weights, self.cvalues), reverse=True)]

  @staticmethod
  def _calcWeights(raw_weights):
    """
    Calculates the weights for values in the categorical collection.
    Notes: Inputs - self.raw_weights
    """
    weights = []
    for xv in zip(*raw_weights):
      weights.append(max([np.abs(x) for x in xv]))
    return weights

  def compareOverlap(self, other, topN=None):
    """
    Calculates the overlap between two CategoricalCollections.
    Computes the ratio of the size of intersection to the size
    of the union.
    """
    if topN is None:
      topN1 = len(self.cvalues)
      topN2 = len(other.cvalues)
    else:
      topN1 = topN
      topN2 = topN
    set1 = set(self.cvalues[0:topN1])
    set2 = set(other.cvalues[0:topN2])
    num_intersection = len(set1.intersection(set2))
    num_union = len(set1.union(set2))
    return (1.0*num_intersection)/num_union
    
  def compareOrder(self, other, top=None):
    """
    Calculates the similarities in ordering of two
    CategoricalCollection.
    """
    pass
