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
"""

import common_python.constants as cn
from common_python.classifier import util_classifier
from common_python.util import util

import copy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn


NUM_CROSS_ITER = 50  # Number of cross validations
FEATURE1 = "feature1"
FEATURE2 = "feature2"
SFA = "sfa"
CPC = "cpc"
IPA = "ipa"
METRICS = [SFA, CPC, IPA]


################## FUNCTIONS #####################
def makeFeatureAnalyzers(clf, df_X, ser_y,
      data_path_pat=None, **kwargs):
    """
    Creates a FeaturAnalyzer for each class.
    :param Classifier clf: binary classifier
    :param pd.DataFrame df_X:
        columns: features
        rows: instances
    :param pd.Series ser_y:
        values: plurality of class values
        rows: instancea
    :param Pattern data_path_pat: string pattern
        arguments are metric, class
    :parm dict kwargs: Optional arguments for constructor
    :return dict:
         key: class
         value: FeatureAnalyzer
    """
    classes = list(ser_y.unique())
    analyzer_dct = {}
    for cl in classes:
      dct = {m: data_path_pat % (m, cl)
          for m in METRICS}
      ser_binary = ser_y.apply(lambda v:
          cn.PCLASS if v==cl else cn.NCLASS)
      analyzer_dct[cl] = FeatureAnalyzer(clf, df_X,
          ser_binary, data_path_dct=dct, **kwargs)
    return analyzer_dct

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


################## CLASSES #####################
class FeatureAnalyzer(object):

  def __init__(self, clf, df_X, ser_y,
      num_cross_iter=NUM_CROSS_ITER, is_report=True,
      report_interval=None,
      data_path_dct=None):
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
    self._features = df_X.columns.tolist()
    self._partitions = [
        util_classifier.partitionByState(self._ser_y)
        for _ in range(num_cross_iter)]
    self._report_interval = report_interval
    if data_path_dct is None:
       data_path_dct = {}
    self._data_path_dct = data_path_dct
    # Number procesed since last report
    self._num_processed = 0
    # Single feature accuracy
    self._ser_sfa = None
    # classifier prediction correlation
    self._df_cpc = None
    # incremental prediction accuracy
    self._df_ipa = None
    # Feature sets accuracies
    self._ser_fea_acc = None

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

  @property
  def ser_sfa(self):
    """
    Construct series for single feature accuracy
      index: feature
      value: accuracy in [0, 1]
    """
    if self._ser_sfa is None:
      if SFA in self._data_path_dct:
        self._ser_sfa = self._readSFA()
      else:
        total = len(self._features)
        self._num_processed = 0
        scores = []
        for feature in self._features:
          df_X = pd.DataFrame(self._df_X[feature])
          score = util_classifier.binaryCrossValidate(
              self._clf, df_X, self._ser_y,
              partitions=self._partitions)
          scores.append(score)
          self._reportProgress(SFA, len(scores), total)
        self._ser_sfa = pd.Series(
            scores, index=self._features)
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
      if CPC in self._data_path_dct.keys():
        self._df_cpc = self._readCPC()
      else:
        total = (len(self._features))**2
        dct = {FEATURE1: [], FEATURE2: [], cn.SCORE: []}
        for feature1 in self._features:
          for feature2 in self._features:
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

  @property
  def df_ipa(self):
    """
    creates pd.DataFrame
        row index: features
        columns: features
        scores: incremental accuracy
    """
    def predict(features):
      df_X = pd.DataFrame(self._df_X[features])
      return util_classifier.binaryCrossValidate(
          self._clf, df_X, self._ser_y,
          partitions=self._partitions)
    #
    if self._df_ipa is None:
      if IPA in self._data_path_dct:
        self._df_ipa = self._readIPA()
      else:
        total = (len(self._features))**2
        dct = {FEATURE1: [], FEATURE2: [], cn.SCORE: []}
        for feature1 in self._features:
          for feature2 in self._features:
            score1 = predict([feature1])
            score2 = predict([feature2])
            score3 = predict([feature1, feature2])
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
  def ser_fea_acc(self):
    """
    Calculates the classification accuracy of feature sets
    of size 1 and 2.
    :return pd.Series: Sorted by descending accuracy
        Index: feature set (feature separated by "+")
        Value: Accuracy
    """
    if self._ser_fea_acc is None:
      # Feature sets of size 1
      ser1 = self.ser_sfa.copy()
      # Feature sets of size two
      feature_sets = []
      accuracies = []
      for idx, feature1 in enumerate(self._features):
        for feature2 in self._features[(idx+1):]:
          feature_sets.append(
              self._makeFeatureSet(feature1, feature2))
          accuracy = self.df_ipa.loc[feature1, feature2] +  \
              max(self._ser_sfa.loc[feature1],
                  self._ser_sfa.loc[feature2])
          accuracies.append(accuracy)
      ser2 = pd.Series(accuracies, index=feature_sets)
      # Construct result
      self._ser_fea_acc = pd.concat([ser1, ser2])
      self._ser_fea_acc = self._ser_fea_acc.sort_values(
          ascending=False)
    return self._ser_fea_acc

  def getMetric(self, metric):
    if metric == SFA:
      return self.ser_sfa
    elif metric == CPC:
      return self.df_cpc
    elif metric == IPA:
      return self.df_ipa
    else:
      raise ValueError("Invalid metric: %s" % metric)

  def readMetric(self, metric, path=None):
    if metric == SFA:
      return self._readSFA(path=path)
    elif metric == CPC:
      return self._readCPC(path=path)
    elif metric == IPA:
      return self._readIPA(path=path)
    else:
      raise ValueError("Invalid metric: %s" % metric)

  def writeMetrics(self, data_path_dct):
    """
    Writes SFA, CPC, IPA to the path indicated.
    :param dict data_path_dct:
      key: in METRICS
      value: str (path)
    """
    data_value_dct = {
        SFA: self.ser_sfa,
        CPC: self.df_cpc,
        IPA: self.df_ipa,
        }
    for key in METRICS:
      data_value_dct[key].to_csv(data_path_dct[key])

  def _readSFA(self, path=None):
    if path is None:
      path = self._data_path_dct[SFA]
    df = pd.read_csv(path)
    columns = df.columns.tolist()
    ser = df[columns[1]]
    ser.index = df[columns[0]]
    ser.name = None
    ser.index.name = None
    return ser.sort_values(ascending=False)

  def _readCPC(self, path=None):
    if path is None:
      path = self._data_path_dct[CPC]
    df = pd.read_csv(self._data_path_dct[CPC])
    return df.set_index(FEATURE1)

  def _readIPA(self, path=None):
    if path is None:
      path = self._data_path_dct[IPA]
    df = pd.read_csv(self._data_path_dct[IPA])
    return df.set_index(FEATURE1)

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

  def _makeFeatureSet(self, feature1, feature2):
    return "%s+%s" % (feature1, feature2)

  def plotCPC(self, **kwargs):
    self._plotHeatmap(CPC, **kwargs)

  def plotIPA(self, **kwargs):
    self._plotHeatmap(IPA, **kwargs)


