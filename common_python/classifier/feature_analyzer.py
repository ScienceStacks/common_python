'''Calculates various statistics for feature selection'''

"""
Single feature accuracy (SFA). Accuracy of a classifier
using a single feature. We denote the accuracy
of a classifier with the single feature F by A(F).

Classifier prediction correlation (CPC). Correlation of
the predictions of two single feature classifiers.
Let P(F) be the predictions produced by a classifier
with just the feature F. Then CPC(F1, F2) is
corr(P(F1), P(F2)).

Incremental prediction accuracy (IPA). Increase in
classification accuracy by using two features in
combination instead of the most accurate of a
two feature classifier.
IPA(F1, F2) = A(F1, F2) - max(A(F1), A(F2))

This module can be run as a main program to recalculate
statistics as follows:
  1. The argument is the absolute path of a serialization
  directory where previous data have been stored. The directory
  must include: df_X.csv (feature vector), ser_y.csv (binary
  class values), and misc.pcl (other constructor objects).
  2. Delete all serialized metric objects that need to be
  recalculated.
  3. Run python feature_analyzer.py <absolute directory path>
"""

import common_python.constants as cn
from common_python.classifier import util_classifier
from common_python.util import util
from common_python.util.persister import Persister

import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn


NUM_CROSS_ITER = 50  # Number of cross validations
FEATURE1 = "feature1"
FEATURE2 = "feature2"
SFA = "sfa"
CPC = "cpc"
IPA = "ipa"
FSET = "fset"
DF_X = "df_x"
SER_Y = "ser_y"
METRICS = [SFA, CPC, IPA, FSET]
VARIABLES = [DF_X, SER_Y]
VARIABLES.extend(list(METRICS))
MISC = "misc"


################## FUNCTIONS #####################
def makeFsetStr(features):
  """
  Creates a string for a feature set
  :param iterable-str features:
  :return str:
  """
  return cn.FEATURE_SEPARATOR.join(list(features))

def unMakeFsetStr(fset_stg):
  """
  Recovers a feature set from a string
  """
  return set(fset_stg.split(cn.FEATURE_SEPARATOR))

def plotSFA(analyzers, num_feature=10, is_plot=True):
  """
  Plots SFA for all classes.
  :param list-FeatureAnalyzer analyzers:
      one analyzer for each class to plot
  :param int num_feature: features per class plotted
  """
  num_classes = len(analyzers)
  fig, ax = plt.subplots(1, num_classes)
  fig.set_figheight(6)
  fig.set_figwidth(18)
  for cl in range(num_classes):
    analyzer = analyzers[cl]
    this_ax = ax[cl]
    xv = analyzer.ser_sfa.index.tolist()[:num_feature]
    yv = analyzer.ser_sfa.values[:num_feature]
    this_ax.bar(xv, yv)
    this_ax.set_title("%d" % cl )
    this_ax.set_xticklabels(xv, fontsize=14)
    if cl == 0:
      this_ax.set_ylabel("Single Feature Accuracy")
      this_ax.set_ylim([0, 1])
    else:
      this_ax.set_yticklabels([])
    this_ax.set_xticklabels(xv, rotation='vertical')
    this_ax.set_ylim([0.48, 1])
    this_ax.yaxis.set_ticks_position('both')
  if is_plot:
    plt.show()

def deserialize(dir_path_dct):
  """
  Creates an analyzer for each class.
  Parameters
  ----------
  dir_path_dct : dict
    key: class
    value: Path to directory for class.

  Returns
  -------
  dict. key: class; value: FeatureAnalyzer
  """
  return {c: FeatureAnalyzer.deserialize(
    dir_path_dct[c]) for c in dir_path_dct.keys()}

def reserialize(dir_path_dct, out_dir_path_dct=None):
  """
  Deserializes the FeatureAnalyzes, re-does the calculations,
  and then serializes.

  Parameters
  ----------
  dir_path_dct : dict
    key: class
    value: Path to read the inputs for deserialization
  out_dir_path_dct : dict
    key: class
    value: Path to write the outputs for deserialization

  Returns
  -------
  None.

  """
  if out_dir_path_dct is None:
    out_dir_path_dct = dict(dir_path_dct)
  analyzer_dct = deserialize(dir_path_dct)
  for idx, analyzer in analyzer_dct.items():
    analyzer.serialize(out_dir_path_dct[idx])


################## CLASSES #####################
class FeatureAnalyzer(object):

  def __init__(self, clf, df_X, ser_y,
      num_cross_iter=NUM_CROSS_ITER, is_report=True,
      report_interval=None):
    """
    :param Classifier clf: binary classifier
    :param pd.DataFrame df_X:
        columns: features
        rows: instances
    :param pd.Series ser_y:
        values: 0, 1
        rows: instances
    :parm int num_cross_iter: iterations in cross valid
    :param int report_interval: number processed
    :param dict data_path_dct: paths for metrics data
       key: SFA, CPC, IPA
       value: path
    """
    ######### PRIVATE ################
    self._clf = copy.deepcopy(clf)
    self._df_X = df_X
    self._ser_y = ser_y
    self._is_report = is_report
    self._num_cross_iter = num_cross_iter
    self._partitions = [
        util_classifier.partitionByState(self._ser_y)
        for _ in range(self._num_cross_iter)]
    self._report_interval = report_interval
    # Number procesed since last report
    self._num_processed = 0
    # Single feature accuracy
    self._ser_sfa = None
    # classifier prediction correlation
    self._df_cpc = None
    # incremental prediction accuracy
    self._df_ipa = None
    # Feature sets accuracies
    self._ser_fset = None
    ######### PUBLIC ################
    self.features = df_X.columns.tolist()

  def _reportProgress(self, metric, count, total):
    """
    Reports progress on computations.
    :param str metric:
    :param int count: how much completed
    :param int total: how much total
    """
    if self._report_interval is not None:
      if count == 0:
        self._num_processed = 0
      elif count - self._num_processed >=  \
          self._report_interval:
        self._num_processed = count
        if self._is_report:
          print("\n***Progress for %s: %d/%d" %
              (metric, count, total))
      else:
        pass

  ### METRICS ###
  @property
  def ser_sfa(self):
    """
    Construct series for single feature accuracy
      index: feature
      value: accuracy in [0, 1]
    """
    if self._ser_sfa is None:
      total = len(self.features)
      self._num_processed = 0
      scores = []
      for feature in self.features:
        df_X = pd.DataFrame(self._df_X[feature])
        score = util_classifier.binaryCrossValidate(
            self._clf, df_X, self._ser_y,
            partitions=self._partitions)
        scores.append(score)
        self._reportProgress(SFA, len(scores), total)
      self._ser_sfa = pd.Series(
          scores, index=self.features)
    return self._ser_sfa

  @property
  def df_cpc(self):
    """
    creates classifier predicition correation pd.DataFrame
        row index: features
        columns: features
        scores: correlation
    """
    if self._df_cpc is None:
      total = (len(self.features))**2
      dct = {FEATURE1: [], FEATURE2: [], cn.SCORE: []}
      for feature1 in self.features:
        for feature2 in self.features:
          clf_desc1 =  \
              util_classifier.ClassifierDescription(
              clf=self._clf, features=[feature1])
          clf_desc2 =  \
              util_classifier.ClassifierDescription(
              clf=self._clf, features=[feature2])
          score = util_classifier.correlatePredictions(
              clf_desc1, clf_desc2, self._df_X,
              self._ser_y, self._partitions)
          dct[FEATURE1].append(feature1)
          dct[FEATURE2].append(feature2)
          dct[cn.SCORE].append(score)
          self._reportProgress(CPC,
              len(dct[FEATURE1]), total)
      df = pd.DataFrame(dct)
      self._df_cpc = df.pivot(index=FEATURE1,
          columns=FEATURE2, values=cn.SCORE)
    return self._df_cpc

  def score(self, features):
      df_X = pd.DataFrame(self._df_X[features])
      return util_classifier.binaryCrossValidate(
          self._clf, df_X, self._ser_y,
          partitions=self._partitions)

  @property
  def df_ipa(self):
    """
    creates pd.DataFrame
        row index: features
        columns: features
        scores: incremental accuracy
    """
    #
    if self._df_ipa is None:
      total = (len(self.features))**2
      dct = {FEATURE1: [], FEATURE2: [], cn.SCORE: []}
      for feature1 in self.features:
        for feature2 in self.features:
          score1 = self.score([feature1])
          score2 = self.score([feature2])
          score3 = self.score([feature1, feature2])
          score = score3 - max(score1, score2)
          dct[FEATURE1].append(feature1)
          dct[FEATURE2].append(feature2)
          dct[cn.SCORE].append(score)
          self._reportProgress(CPC,
              len(dct[FEATURE1]), total)
          self._reportProgress(IPA,
              len(dct[FEATURE1]), total)
      df = pd.DataFrame(dct)
      self._df_ipa = df.pivot(index=FEATURE1,
          columns=FEATURE2, values=cn.SCORE)
    return self._df_ipa

  @property
  def ser_fset(self):
    """
    Calculates the classification accuracy of feature sets
    of size 1 and 2.
    :return pd.Series: Sorted by descending accuracy
        Index: feature set (feature separated by "+")
        Value: Accuracy
    """
    if self._ser_fset is None:
      # Feature sets of size 1
      ser1 = self.ser_sfa.copy()
      # Feature sets of size two
      feature_sets = []
      accuracies = []
      for idx, feature1 in enumerate(self.features):
        for feature2 in self.features[(idx+1):]:
          feature_sets.append(makeFsetStr(
              [feature1, feature2]))
          try:
            accuracy = self.df_ipa.loc[feature1, feature2] +  \
                max(self.ser_sfa.loc[feature1],
                    self.ser_sfa.loc[feature2])
          except KeyError:
            accuracy = np.nan  # Handle missing keys
          accuracies.append(accuracy)
      ser2 = pd.Series(accuracies, index=feature_sets)
      # Construct result
      self._ser_fset = pd.concat([ser1, ser2])
      self._ser_fset = self._ser_fset.sort_values(
          ascending=False)
    return self._ser_fset

  def getMetric(self, metric):
    if metric == SFA:
      return self.ser_sfa
    elif metric == CPC:
      return self.df_cpc
    elif metric == IPA:
      return self.df_ipa
    elif metric == FSET:
      return self.ser_fset
    else:
      raise ValueError("Invalid metric: %s" % metric)


  ### PLOTS ###

  def _plotHeatmap(self, metric, is_plot=True,
                   title=None, **kwargs):
    """
    Heatmap plot.
    :param str metric: metric to plot
    :param bool is_plot: show plot
    :param str title:
    :param dict kwargs: passed to pruneValues
    """
    if title is None:
      title = metric
    df = util.trimDF(self.getMetric(metric),
                               **kwargs)
    df.index.name = None
    if len(df) > 1:
      _ = seaborn.clustermap(df, col_cluster=True,
          row_cluster=True,
          xticklabels=True, yticklabels=True,
          vmin=-1, vmax=1,
          cbar_kws={"ticks":[-1, -0.5, 0, 0.5, 1]},
          cmap="seismic")
      plt.title(title)
      if is_plot:
        plt.show()
      return df
    else:
      return None

  def plotCPC(self, **kwargs):
    self._plotHeatmap(CPC, **kwargs)

  def plotIPA(self, **kwargs):
    self._plotHeatmap(IPA, **kwargs)

  #### SERIALIZE and DESERAIALIZE ###

  @classmethod
  def _getPath(cls, dir_path, name, ext="csv"):
    """
    Constructs a path for the variable name.
    Creates directory path if it does not exist.
    Parameters
    ----------
    cls : Type
    dir_path: str
      directory path
    name : str
      Variable name.
    ext: str
      File extension

    Returns
    -------
    str. File path

    """
    if not os.path.isdir(dir_path):
      os.mkdir(dir_path)
    path = os.path.join(dir_path,
                        "%s.%s" % (name, ext))
    return path

  @staticmethod
  def _makeSer(df, is_sort=True):
    """
    Converts a DataFrame representation of a Series
    into a Series.
    :return pd.Series:
    """
    if df is None:
      return df
    columns = df.columns.tolist()
    ser = df[columns[1]]
    ser.index = df[columns[0]]
    ser.name = None
    ser.index.name = None
    if is_sort:
      ser = ser.sort_values(ascending=False)
    return ser

  @staticmethod
  def _makeMatrix(df):
    """
    Common post-processing of dataframe deserialization of
    metrics.
    :param pd.DataFrame df:
    """
    if df is None:
      return df
    else:
      return df.set_index(FEATURE1)

  def serialize(self, dir_path):
    """
    Writes all state to a directory.
    Parameters
    ----------
    dir_path : str
      DESCRIPTION. Path to where data are written.

    Returns
    -------
    None.

    """
    if not os.path.isdir(dir_path):
      os.mkdir(dir_path)
    values = [self.ser_sfa, self.df_cpc, self.df_ipa,
              self.ser_fset, self._df_X, self._ser_y]
    variables = [SFA, CPC, IPA, FSET, DF_X, SER_Y]
    for idx, name in enumerate(variables):
      path = FeatureAnalyzer._getPath(dir_path, name)
      values[idx].to_csv(path)
    # Serialize the classifier
    path = FeatureAnalyzer._getPath(dir_path, MISC, ext=cn.PCL)
    persister = Persister(path)
    persister.set([self._clf, self._num_cross_iter,
                    self._is_report, self._report_interval])

  @classmethod
  def deserialize(cls, dir_path):
    """
    Read data from directory.
    Note must be maintained in correspondence with serialize,
    especially for values written the MISC.

    Parameters
    ----------
    dir_path : str
      DESCRIPTION. Directory path

    Returns
    -------
    FeatureAnalyzer

    """
    def readDF(name):
      path = FeatureAnalyzer._getPath(dir_path, name)
      df = pd.read_csv(path)
      return df
    #
    # Obtain non-metric values
    path = FeatureAnalyzer._getPath(dir_path, MISC, ext=cn.PCL)
    persister = Persister(path)
    [clf, num_cross_iter, is_report, report_interval] =  \
        persister.get()
    df_X = util.trimUnnamed(readDF(DF_X))
    ser_y = FeatureAnalyzer._makeSer(readDF(SER_Y),
                                     is_sort=False)
    # Instantiate
    analyzer = cls(clf, df_X, ser_y,
                   num_cross_iter=num_cross_iter,
                   is_report=is_report,
                   report_interval=report_interval)
    # Set values of metrics
    for metric in METRICS:
      path = FeatureAnalyzer._getPath(dir_path, metric)
      try:
        df = pd.read_csv(path)
      except FileNotFoundError:
        # Indicate not present
        df = None
      if metric == SFA:
        analyzer._ser_sfa = FeatureAnalyzer._makeSer(df)
      elif metric == CPC:
        try:
          analyzer._df_cpc = FeatureAnalyzer._makeMatrix(df)
        except:
          pass
      elif metric == IPA:
        analyzer._df_ipa = FeatureAnalyzer._makeMatrix(df)
      elif metric == FSET:
        analyzer._ser_fset = FeatureAnalyzer._makeSer(df)
    #
    return analyzer


if __name__ == '__main__':
  msg = "Run FeatureAnalyzer for for a directory path."
  parser = argparse.ArgumentParser(description=msg)
  parser.add_argument("path",
      help="Absolute path to the serialization directory",
      type=str)
  args = parser.parse_args()
  reserialize({1: args.path})







