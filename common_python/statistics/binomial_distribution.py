"""Tests based on the binomial distribution."""

"""
Provides calculations for significance levels for Binomial processes.
A positive (pos) event is the occurrence of something counted.
A negative (neg) event is the non-occurrence. Positive plus negative
events equals the total sample size.
"""


import common_python.constants as cn

import pandas as pd
import numpy as np
import scipy.stats as stats


BINOMIAL_PROB = 0.5


class BinomialDistribution(object):

  def __init__(self, max_count, binomial_prob=BINOMIAL_PROB):
    """
    Parameters
    ----------
    max_count: int
        Maximum value for the number in the binomial population
    binomial_prob: float
    """
    self.max_count = max_count
    self.binomial_prob = binomial_prob
    # Matrix of significance levels for having at least n events
    #   row: sample_size
    #   col: number of events
    # Matrix for positive events
    self.pos_sl_mat = self._populateSignificanceLevels(self.binomial_prob)
    # Matrix for negative events
    self.neg_sl_mat = self._populateSignificanceLevels(1 - self.binomial_prob)

  def _populateSignificanceLevels(self, prob):
    """
    Populates the signifiance level matrices.

    Parameters
    ----------
    prob: float
    
    Returns
    -------
    matrix
        rows are sample size
        columns are number events
    """
    def calcTailProb(sample_count, nval):
      if nval == 0:
        return 1.0
      return 1 - stats.binom.cdf(nval - 1, sample_count, prob)
    # Initialize the matrix 
    size = self.max_count + 1
    mat = np.repeat(np.nan, size*size)
    mat = np.reshape(mat, (size, size))
    #
    for sample_count in range(self.max_count + 1):
      for npos in range(sample_count + 1):
        mat[sample_count, npos] = calcTailProb(sample_count, npos)
    #
    return mat

  def getSL(self, num_sample, num_pos_event, is_two_sided=True):
    """
    Calculates the significance level of obtaining at least num_pos_event's out of
    num_sample. If the test is two sided, then it also calculates
    the significance of num_pos_event's or fewer. In a two sided tests, the
    smaller of the two significance levels is returned. If the smaller one
    is for "too few events", the value is negative.

    Parameters
    ----------
    num_sample: int
    num_pos_event: int
    is_two_sided: bool
    
    Returns
    -------
    float
    """
    pos_sl = self.pos_sl_mat[num_sample, num_pos_event]
    num_neg_event = num_sample - num_pos_event
    neg_sl = self.neg_sl_mat[num_sample, num_neg_event]
    if pos_sl < neg_sl:
      sl = pos_sl
    else:
      sl = -neg_sl
    return sl

  def isLowSL(self, num_sample, num_event, max_sl, **kwargs):
    """
    Tests if the significance level is no larger than max_sl.

    Parameters
    ----------
    num_sample: int
    num_event: int
    max_sl: float
    kwargs: optional arguments passed to getSL
    
    Returns
    -------
    bool
    """
    return np.abs(self.getSL(num_sample, num_event, **kwargs)) <= max_sl
