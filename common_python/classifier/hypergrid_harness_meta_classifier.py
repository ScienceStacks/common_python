"""HypergridHarness for MetaClassifiers."""

import common_python.constants as cn
from common_python.classifier.hypergrid_harness  \
    import HypergridHarness
import common_python.util.util as util
from common_python.util.item_aggregator import ItemAggregator
from common_python.plots.plotter import Plotter

import collections
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##################### FUNCTIONS ###################
def _assignObjectValues(target, source):
  """
  Assigns the values in the source to the target.
  Source and target must be the same type.
  :param object target:
  :param object source:
  """
  for key, value in source.__dict__.items():
    target.__dict__[key] = value


#################### CLASSES ######################

ScoreResults = collections.namedtuple("ScoreResults",
    "abs rel")


class HypergridHarnessMetaClassifier(HypergridHarness):

  # Values are dataframes with the columns cn.MEAN, cn.STD
  # and rows are MetaClassifiers evaluated.

  def __init__(self, mclfs, impurity=0, **kwargs):
    """
    :param list-MetaClassifier mclfs: to be studied
    :param float impurity: in [-1, 1]
    :param dict kwargs: arguments in HypergridHarness constructor
    """
    self.mclfs = mclfs
    if impurity == 0:
      harness = HypergridHarness(**kwargs)
    else:
      #harness = HypergridHarness.initImpure(impurity, **kwargs)
      raise RuntimeError("Impurity != 0 isn't implemented.")
    # Copy all data to the new to this harness
    _assignObjectValues(self, harness)

  def _evaluateExperiment(self, sigma=0, num_repl=1):
    """
    Evaluates the classification accuracy of MetaClassifiers
    for a single experiment.
    :param float sigma:
    :param int num_repl: Number of replications passed to classifiers
    :return list-ScoreResult:
    """
    train_trinarys = self.trinary.perturb(sigma=sigma, num_repl=num_repl)
    test_trinary = self.trinary.perturb(sigma=sigma, num_repl=1)[0]
    dfs = [trinary.df_feature for trinary in train_trinarys]
    [m.fit(dfs, self.trinary.ser_label) for m in selfmclfs]
    score_results = [
        m.score(test_trinary.df_feature, self.trinary.ser_label)
        for m in self.mclfs]
    return score_results

  def evaluate(self, count=10, sigmas=[0], num_repls=[1]):
    """
    Evaluates the classification accuracy of MetaClassifier
    for different conditions. Each experiment is repeated several times.
    :param list-float sigmas:
    :param list-int num_repls:
    :return dict:
         key: (sigma,num_repl)
         value: ScoreResults
    """
    #
    result = {}
    for sigma in sigmas:
      for num_repl in num_repls:
        aggregator_abs = ItemAggregator(lambda s: s.abs)
        aggregator_rel = ItemAggregator(lambda s: s.rel)
        for _ in range(count):
          key = (sigma, num_repl)
          results = self._evaluateExperiment(sigma=sigma,
              num_repl=num_repl)
          aggregator_abs.append(results)
          aggregator_rel.append(results)
        #
        df_abs = aggregator_abs.df
        df_rel = aggregator_rel.df
        result[(sigma, num_repl)] = ScorerResults(
            abs=df_abs, rel=df_rel)
    #
    return result