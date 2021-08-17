"""Tests based on the binomial distribution."""


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
    self._sl_mat = None

  def _populateSignificanceLevels(self):
    """
    Populates the signifiance level matrices.
    """
    def calcTailProb(sample_count, nval):
      if nval == 0:
        return 1.0
      return 1 - stats.binom.cdf(nval - 1, sample_count, self.binomial_prob)
    # Initialize the matrix 
    size = self.max_count + 1
    mat = np.repeat(np.nan, size*size)
    self._sl_mat = np.reshape(mat, (size, size))
    #
    for sample_count in range(self.max_count + 1):
      for npos in range(sample_count + 1):
        self._sl_mat[sample_count, npos] = calcTailProb(sample_count, npos)

  @property
  def sl_mat(self):
    if self._sl_mat is None:
      self._populateSignificanceLevels()
    return self._sl_mat

  def getSL(self, num_sample, num_event, is_two_sided=True):
    """
    Calculates the significance level of obtaining at least num_event's out of
    num_sample. If the test is two sided, then it also calculates
    the significance of num_event's or fewer. In a two sided tests, the
    smaller of the two significance levels is returned. If the smaller one
    is for "too few events", the value is negative.

    Parameters
    ----------
    num_sample: int
    num_event: int
    is_two_sided: bool
    
    Returns
    -------
    float
    """
    too_many_prob = self.sl_mat[num_sample, num_event]
    num_non_event = num_sample - num_event
    too_few_prob = self.sl_mat[num_sample, num_non_event]
    if too_many_prob < too_few_prob:
      sl = too_many_prob
    else:
      sl = -too_few_prob
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
