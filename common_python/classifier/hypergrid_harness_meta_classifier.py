"""HypergridHarness for MetaClassifiers."""

import common_python.constants as cn
from common_python.experiment.experiment_harness import ExperimentHarness
from common_python.classifier.hypergrid_harness  \
    import Vector, Plane
from common_python.classifier.random_hypergrid_harness  \
    import RandomHypergridHarness
from common_python.classifier.meta_classifier  \
    import MetaClassifierDefault, MetaClassifierPlurality,  \
    MetaClassifierAugment, MetaClassifierAverage, \
    MetaClassifierEnsemble
import common_python.util.util as util
from common_python.util.item_aggregator import ItemAggregator
from common_python.plots.plotter import Plotter

import os
import collections
import copy
import matplotlib.pyplot as plt
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
PTH = os.path.join(cn.CODE_DIR, "classifier")
EVALUATION_DATA_PTH = os.path.join(PTH,
    "hypergrid_harness_meta_classifier.csv")
POLICY = "policy"
SIGMA = "sigma"
IMPURITY = "impurity"
NUM_DIM = "num_dim"


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


class HypergridHarnessMetaClassifier(RandomHypergridHarness):

  # Values are dataframes with the columns cn.MEAN, cn.STD
  # and rows are MetaClassifiers evaluated.

  def __init__(self, mclf_dct=MCLF_DCT, **kwargs):
    """
    :param dict mclf_dct: classifiers to study to be studied
    :param dict kwargs: arguments in HypergridHarness constructor
    """
    super().__init__(**kwargs)
    self.mclf_dct = mclf_dct
    # Get previously analyzed data
    if os.path.isfile(EVALUATION_DATA_PTH):
      self.df_data = pd.read_csv(EVALUATION_DATA_PTH)
    else:
      self.df_data = None

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

  @classmethod
  def analyze(cls, mclf_dct=MCLF_DCT, num_repl=3,
      sigma=1.5, num_dim=5,
      iter_count=ITER_COUNT, is_rel=True,
      is_iter_report=True,
      **kwargs):
    """
    Compares multiple policies for handling feature replications.
    :param dict mclf_dct: dictionary of MetaClassifer
    :param int num_repl: Number of replications of feature vectors
    :param float sigma: std of perturbation of features
    :param int num_dim: dimension of the hypergrid space
    :param bool is_rel: report relative scores
    :parm int iter_count: number of iterations to calculate statistics
    :param bool is_iter_report: report on each iteration
    :param dict kwargs: arguments to RandomHypergridHarness
    :return pd.DataFrame: columns
        POLICY, cn.MEAN, cn.STD, cn.COUNT
    """
    if is_rel:
      sel_func = lambda v: v.rel
    else:
      sel_func = lambda v: v.abs
    #
    if not IMPURITY in kwargs.keys():
      kwargs[IMPURITY] = 0
    harness = HypergridHarnessMetaClassifier(
        mclf_dct=mclf_dct, **kwargs)
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
          % (sigma, num_dim, kwargs[IMPURITY], iter_count))
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

  @classmethod
  def makeEvaluationData(cls, is_test=False,
      out_pth=EVALUATION_DATA_PTH):
    """
    Generate data evaluating meta-classifiers on a hypergrid.
    :param bool is_test: Test invocation
    :param str out_pth: path to file written
    """
    def posToImpurity(pos_frac):
      return np.round(2*pos_frac - 1, 2)
    def runner(sigma=None, num_dim=None, impurity=None,
        iter_count=1000):
      if is_test:
        iter_count = 2
        is_iter_report = False
      else:
        is_iter_report = True
      return HypergridHarnessMetaClassifier.analyze(
          mclf_dct=MCLF_DCT,
          sigma=sigma, num_dim=num_dim, 
          iter_count=iter_count,
          is_iter_report = is_iter_report,
          num_repl=3, is_rel=False, 
          # RandomHypergridHarness arguments
          impurity=impurity, num_point=25, density=10)
    if not is_test:
      param_dct = {
          SIGMA: [0, 0.2, 0.5, 1.0, 1.5, 2.0],
          IMPURITY: [0, 
          posToImpurity(2/25),
          posToImpurity(3/25),
          posToImpurity(4/25),
          posToImpurity(5/25),
          posToImpurity(6/25),
          ],
          NUM_DIM: [2, 5, 7, 10, 15, 20, 25, 30],
          }
    else:
      param_dct = {
          SIGMA: [0],
          IMPURITY: [0, 
          posToImpurity(6/25),
          ],
          NUM_DIM: [2],
          }
    harness = ExperimentHarness(param_dct, runner, update_rpt=1,
        out_pth=out_pth)
    harness.run()
    if not is_test:
      print("Done processing.")

  def plotMetaClassifiers(self, num_dim, impurity, ax=None, **kwargs):
    """
    Plots the meta-classifiers from the evaluation data.
    x-axis: sigma
    y-axis: accuaracy
    :param int num_dim:
    :param float impurity:
    :param dict kwargs: optional plot arguments
    """
    if ax is None:
      plotter = Plotter()
      ax = plotter.ax
    else:
      plotter = None
    sel = [(r[NUM_DIM]==num_dim) and (r[IMPURITY]==impurity)
        for _, r in self.df_data.iterrows()]
    df = self.df_data.loc[sel, :]
    df = df[[POLICY, cn.MEAN, SIGMA, cn.STD]]
    policies = df[POLICY].unique()
    for policy in policies:
      df_plot = df[df[POLICY] == policy]
      ax.errorbar(df_plot[SIGMA],
           df_plot[cn.MEAN], 2*df_plot[cn.STD], label=policy)
    if plotter is not None:
      plotter.ax.legend()
      title = "num_dim: %d, impurity: %2.2f" % (num_dim, impurity)
      plotter.do(xlabel="std", ylabel="accuracy", title=title,
         xlim=[0, 2], ylim=[0.5, 1.0])
         # xlim=[0, 2], ylim=[0.5, 1.0], legend=policies)

  def plotMultipleMetaClassifiers(self, num_dim, impuritys, **kwargs):
    """
    Plots the meta-classifiers from the evaluation data.
    x-axis: sigma
    y-axis: accuaracy
    :param int num_dim:
    :param list-float impurity:
    :param dict kwargs: optional plot arguments
    """
    subplots = []
    length = len(impuritys)
    for idx in range(1, length + 1):
      subplots.append((1, length, idx))
    plotter = Plotter(subplots=subplots)
    policies = [p for p in self.df_data[POLICY].unique()]
    for idx, impurity in enumerate(impuritys):
      pltargs = {}
      if idx == 0:
        pltargs[cn.PLT_YLABEL] = "accuracy"
      else:
        pltargs[cn.PLT_YLABEL] = ""
        pltargs[cn.PLT_YTICKLABELS] = ""
      if idx == length - 1:
        pltargs[cn.PLT_LEGEND] = policies
      pltargs[cn.PLT_TITLE] = "%2.2f" % impurity
      pltargs[cn.PLT_XLABEL] = "std"
      pltargs[cn.PLT_XLIM] = [0, 2.0]
      pltargs[cn.PLT_YLIM] = [0.5, 1.0]
      plotter.doAx(plotter.axes[idx], **pltargs)
      self.plotMetaClassifiers(num_dim, impurity,
          ax=plotter.axes[idx])
    plotter.axes[-1].legend(loc="upper right")
    plotter.do()
  

if __name__ == '__main__':
  HypergridHarnessMetaClassifier.makeEvaluationData()
