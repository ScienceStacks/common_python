"""HypergridHarness for MetaClassifiers."""

import common_python.constants as cn
from common_python.classifier.hypergrid_harness  \
    import HypergridHarness
from common_python.classifier.meta_classifier  \
    import MetaClassifierDefault, MetaClassifierPlurality,  \
    MetaClassifierAugment, MetaClassifierAverage, \
    MetaClassifierEnsemble
import common_python.util.util as util
from common_python.util.item_aggregator import ItemAggregator
from common_python.plots.plotter import Plotter

import collections
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MAX_ITER = 1000  # Maximum of iterations in analysis
MCLFS = [
    MetaClassifierPlurality(),  # IDX_PLURALITY
    MetaClassifierDefault(),    # IDX_DEFAULT
    MetaClassifierAugment(),    # IDX_AUGMENT
    MetaClassifierAverage(),    # IDX_AVERAGE
    MetaClassifierEnsemble(),   # IDX_ENSEMBLE
    ]
IDX_PLURALITY = 0
IDX_DEFAULT = 1
IDX_AUGMENT = 2
IDX_AVERAGE = 3
IDX_ENSEMBLE = 4


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

  def __init__(self, mclfs=MCLFS, **kwargs):
    """
    :param list-MetaClassifier mclfs: to be studied
    :param dict kwargs: arguments in HypergridHarness constructor
    """
    self.mclfs = mclfs
    harness = HypergridHarness(**kwargs)
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
    [m.fit(dfs, self.trinary.ser_label) for m in self.mclfs]
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

  @classmethod
  def analyze(cls, mclfs=MCLFS, num_repl=3, sigmas=[1.5], num_dim=5,
      is_rel=True, **kwargs):
    """
    Compares multiple polices for handling feature replications.
    :param list-MetaClassifier mclfs: to be studied
    :param int num_repl: Number of replications of feature vectors
    :param list-float sigmas: list of std of perturbation of features
    :param int num_dim: dimension of the hypergrid space
    :param dict kwargs: arguments to HypergridHarness constructor
    :param bool is_rel: report relative scores
    :return pd.Series, pd.Series, int:
        mean, std, number of experiments
    """
    if is_rel:
      sel_func = lambda v: v.rel
    else:
      sel_func = lambda v: v.abs
    #
    vector = Vector(np.repeat(1, num_dim))
    plane = Plane(vector)
    harness = HypergridHarnessMetaClassifier(
        mclfs, density=1.5, plane=plane,
        num_point=num_point, impurity=0)
    for sigma in sigmas:
      scoress = []
      for _ in range(MAX_ITER):
        try:
          score_results = harness._evaluateExperiment(
              sigma=sigma, num_repl=num_repl)
          rel_scores = [sel_func(score_results[i])
              for i in range(len(score_results))]
          scoress.append(rel_scores)
        except:
          pass
      arr = np.array(scoress)
      df = pd.DataFrame(arr)
      ser_mean = df.mean()
      num_exp = len(scoress)
      ser_std = df.std() / np.sqrt(num_exp)
      df = pd.DataFrame({
        cn.MEAN: ser_mean,
        cn.STD: ser_std,
        cn.COUNT: np.repeat(num, len(ser_mean)),
        "sigma": np.repeat(sigma, len(ser_mean)),
        })
     df.append(dfs)
    return ser_mean, ser_std, num_exp
