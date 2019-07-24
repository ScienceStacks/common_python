"""Analyzes collections of categorical values."""

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
    self.terms = terms
    self.raw_weights = raw_weights
    # Choose the weight with the largest absolute value
    self.weights = [xv[argmax([np.abs(x) for x in xv)] 
        for xv in zip(*self.raw_weights)]

  def compareOverlap(self, other, top=None):
    """
    Calculates the overlap between two CategoricalCollections.
    """

  def compareOrder(self, other, top=None):
    """
    Calculates the similarities in ordering of two
    CategoricalCollection.
    """
