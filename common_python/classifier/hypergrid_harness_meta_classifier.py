"""HypergridHarness for MetaClassifiers."""

import common_python.constants as cn
from common_python.experiment.experiment_harness import ExperimentHarness
from common_python.classifier.hypergrid_harness  \
    import HypergridHarness, Vector, Plane
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

ITER_COUNT = 100  # Number of iterations used to calculate statistics
MCLF_DCT = {
    "plurality": MetaClassifierPlurality(),
    "default": MetaClassifierDefault(),
    "augment": MetaClassifierAugment(),
    "average": MetaClassifierAverage(),
    "ensemble": MetaClassifierEnsemble(),
    }
OUT_PATH = "hypergrid_harness_meta_classifier.csv"
POLICY = "policy"


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

  def __init__(self, mclf_dct=MCLF_DCT, **kwargs):
    """
    :param dict mclf_dct: classifiers to study to be studied
    :param dict kwargs: arguments in HypergridHarness constructor
    """
    self.mclf_dct = mclf_dct
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
    [m.fit(dfs, self.trinary.ser_label) for m in self.mclf_dct.values()]
    score_results = [
        m.score(test_trinary.df_feature, self.trinary.ser_label)
        for m in self.mclf_dct.values()]
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
  def analyze(cls, mclf_dct=MCLF_DCT, num_repl=3,
      sigma=1.5, num_dim=5,
      iter_count=ITER_COUNT, is_rel=True,
      is_iter_report=True,
      **kwargs):
    """
    Compares multiple polices for handling feature replications.
    :param dict mclf_dct: dictionary of MetaClassifer
    :param int num_repl: Number of replications of feature vectors
    :param float sigma: std of perturbation of features
    :param int num_dim: dimension of the hypergrid space
    :param bool is_rel: report relative scores
    :parm int iter_count: number of iterations to calculate statistics
    :param bool is_iter_report: report on each iteration
    :param dict kwargs: arguments to HypergridHarness constructor
    :return pd.DataFrame: columns
        POLICY, cn.MEAN, cn.STD, cn.COUNT
    """
    if is_rel:
      sel_func = lambda v: v.rel
    else:
      sel_func = lambda v: v.abs
    #
    if "impurity" in kwargs.keys():
      impurity = kwargs["impurity"]
    else:
      impurity = 0
    vector = Vector(np.repeat(1, num_dim))
    plane = Plane(vector)
    harness = HypergridHarnessMetaClassifier(
        mclf_dct=mclf_dct, plane=plane, **kwargs)
    scoress = []
    dfs = []
    for cnt in range(iter_count):
      try:
        score_results = harness._evaluateExperiment(
            sigma=sigma, num_repl=num_repl)
        rel_scores = [sel_func(score_results[i])
            for i in range(len(score_results))]
        scoress.append(rel_scores)
      except:
        pass
    if is_iter_report:
      print("sigma=%2.2f, num_dim=%d, impurity=%2.2f iter=%d"
          % (sigma, num_dim, impurity, iter_count))
    arr = np.array(scoress)
    df = pd.DataFrame(arr)
    ser_mean = df.mean()
    num_exp = len(scoress)
    ser_std = df.std() / np.sqrt(num_exp)
    df = pd.DataFrame({
        POLICY: list(mclf_dct.keys()),
        cn.MEAN: ser_mean,
        cn.STD: ser_std,
        cn.COUNT: np.repeat(num_exp, len(mclf_dct)),
        })
    return df


if __name__ == '__main__':
  def runner(sigma=None, num_dim=None, impurity=None):
    return HypergridHarnessMetaClassifier.analyze(mclf_dct=MCLF_DCT,
        sigma=sigma, num_dim=num_dim, 
        iter_count=1000,
        num_repl=3, is_rel=False, 
        # HypergridHarness arguments
        impurity=impurity, num_point=25, density=10)
  if True:
    param_dct = {
        "sigma": [0, 0.2, 0.5, 1.0, 1.5, 2.0],
        "impurity": [0, -0.76, -0.6],
        "num_dim": [2, 5, 7, 15, 30],
        }
    harness = ExperimentHarness(param_dct, runner, update_rpt=1,
        out_path=OUT_PATH)
    harness.run()
  print("Done processing.")
