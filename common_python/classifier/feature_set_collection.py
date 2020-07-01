'''Constructs and evaluates alternative classifier features.'''

import common_python.constants as cn
from common_python.classifier.feature_set  \
    import FeatureSet, Case
from common_python.classifier import util_classifier
from common_python.classifier import feature_analyzer
from common_python.util.persister import Persister

import argparse
import collections
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn

MIN_FRAC_INCR = 1.01  # Must increase by at least 1%
MIN_SCORE = 0
SER_SBFSET = "ser_sbfset"
SER_COMB = "ser_comb"
DF_CASE = "df_case"
COMPUTES = [SER_SBFSET, SER_COMB, DF_CASE]
MISC_PCL = "feature_set_collection_misc.pcl"
MIN_SL = 1e-10
MAX_SL = 1

############ FUNCTIONS #################
def disjointify(ser_fset, min_score=MIN_SCORE):
  """
  Makes disjoint the feature sets disjoint by discarding
  feature sets that have non-null intersection with a
  more accurate feature set.

  Parameters
  ----------
  ser_fset: pd.Series
  max_score : float
      minimum classification score

  Returns
  -------
  pd.Series
  """
  ser = ser_fset[ser_fset >= min_score]
  ser = ser.copy()
  selecteds = []  # Features selected
  fset_stgs = []  # Feature strings selected
  for fset_stg in ser.index:
    fset =  FeatureSet(fset_stg)
    if len(fset.set.intersection(selecteds)) > 0:
      continue
    else:
      # Include this feature set
      selecteds.extend(list(fset.set))
      fset_stgs.append(fset.str)
  ser_result = ser[fset_stgs].copy()
  return ser_result


############### CLASSES ####################
class FeatureSetCollection(object):

  def __init__(self, analyzer, min_score=MIN_SCORE):
    """
    Parameters
    ----------
    analyzer: FeatureAnalyzer
    min_score: float
        minimum accuracy score used in calculations
    """
    self._analyzer = analyzer
    self._min_score = min_score
    self._ser_sbfset = None  # Use self.make() to construct
    #  index: string representation of a set of features
    #  value: accuracy (score)
    self._ser_comb = None  # Combinations of feature sets
    self._df_case = None  # Cases for FeatureSet

  @property
  def df_case(self):
    """
    Dataframe describing cases for the FeatureSets in
    ser_comb. A case is an assignment of trinary
    values to the features in a feature set.
    :return pd.Series:
        Columns: cn.FEATURE_SET, cn.CASE, cn.SIGLVL_ZEROES
    Notes:
      1. cn.SIGLVL_ZEROES may be negative to 
         indicate significance level for a negative case.
    """
    return self._makeCase()

  def _makeCase(self): 
    """
    Dataframe describing cases for the FeatureSets in
    ser_comb. A case is an assignment of trinary
    values to the features in a feature set.
    :return pd.Series:
        Columns: cn.FEATURE_SET, cn.CASE, cn.NUM_ZERO
    Notes:
      1. cn.SIGLVL_ZEROES may be negative to 
         indicate significance level for a negative case.
    """
    def convert(siglvl_pos, siglvl_neg):
      """
      Converts significance level to number of zeroes.
      Significance level for a negative class is the
      negative of its number of zeroes.
      """
      def handleNanAndZero(s):
        if np.isnan(s):
          value = MAX_SL
        else:
          if s < MIN_SL:
            value = MIN_SL
          else:
            value = s
        return value
      #
      siglvl_pos = handleNanAndZero(siglvl_pos)
      siglvl_neg = handleNanAndZero(siglvl_neg)
      if siglvl_pos < siglvl_neg:
        num_zero = -np.log10(siglvl_pos)
      elif siglvl_neg == 1:
        num_zero = 0
      else:
        num_zero = np.log10(siglvl_neg)
      return num_zero
    #
    if self._df_case is None:
      dct = {
          cn.FEATURE_SET: [], cn.CASE: [],
          cn.NUM_ZERO: [] }
      for fset_stg in self.ser_comb.index:
        fset = FeatureSet(fset_stg,
            analyzer=self._analyzer)
        df_profile = fset.profileTrinary()
        for case_tuple in df_profile.index:
          siglvl_pos = df_profile.loc[[case_tuple],
               cn.SIGLVL_POS][0]
          siglvl_neg = df_profile.loc[[case_tuple],
               cn.SIGLVL_NEG][0]
          num_zero = convert(siglvl_pos, siglvl_neg)
          dct[cn.FEATURE_SET].append(fset.str)
          dct[cn.CASE].append(case_tuple)
          dct[cn.NUM_ZERO].append(num_zero)
      self._df_case = pd.DataFrame(dct)
    return self._df_case
        
  @property
  def ser_sbfset(self):
    """
    Calculates the classification accuracy of singleton
    and binary feature sets.
    :return pd.Series: Sorted by descending accuracy
        Index: feature set (feature separated by "+")
        Value: Accuracy
    """
    if self._ser_sbfset is None:
      # Feature sets of size 1
      ser1 = self._analyzer.ser_sfa.copy()
      # Feature sets of size two
      feature_sets = []
      accuracies = []
      for idx, feature1 in enumerate(
          self._analyzer.features):
        for feature2 in self._analyzer.features[(idx+1):]:
          fset = FeatureSet([feature1, feature2])
          feature_sets.append(fset.str)
          try:
            accuracy = self._analyzer.df_ipa.loc[
                feature1, feature2] +  \
                max(self._analyzer.ser_sfa.loc[feature1],
                    self._analyzer.ser_sfa.loc[feature2])
          except KeyError:
            accuracy = np.nan  # Handle missing keys
          accuracies.append(accuracy)
      ser2 = pd.Series(accuracies, index=feature_sets)
      # Construct result
      self._ser_sbfset = pd.concat([ser1, ser2])
      self._ser_sbfset = self._ser_sbfset.sort_values(
          ascending=False)
    return self._ser_sbfset

  @property
  def ser_comb(self):
    """
    Optimizes the collection of features sets by
    finding increases in score accuracy.

    Parameters
    ----------

    Returns
    -------
    pd.Series
    """
    def update(fset):
      """
      Refines the feature set and updates data.
      """
      be_result = self._analyzer.backEliminate(
          list(fset.set))
      new_fset = FeatureSet(be_result.sub)
      result_dct[new_fset.str] = be_result.score
      # Put back the features that are eliminated
      for feature in be_result.elim:
        score = self._analyzer.ser_sfa.loc[feature]
        if score >= self._min_score:
          process_dct[feature] = score
      return
    #
    if self._ser_comb is None:
      import pdb; pdb.set_trace()
      ser = self.disjointify(min_score=self._min_score)
      process_dct = ser.to_dict()
      result_dct = {}
      #
      def getScore(fset):
        # Gets the score for an fset
        return process_dct[fset.str]
      # Iteratively consider combinations of fsets
      while len(process_dct) > 0:
        cur_fset = FeatureSet(list(process_dct.keys())[0])
        cur_score = process_dct[cur_fset.str]
        if len(process_dct) == 1:
          if cur_score >= self._min_score:
            update(cur_fset)
          if len(process_dct) <= 2:
            del process_dct[cur_fset.str]
            break
        #
        del process_dct[cur_fset.str]
        # Look for a high accuracy feature set
        is_changed = False
        for other_fset_stg in process_dct.keys():
          other_fset = FeatureSet(other_fset_stg)
          new_fset = FeatureSet(
              cur_fset.set.union(other_fset.set))
          new_score = self._analyzer.score(new_fset.set)
          old_score =  max(cur_score, getScore(other_fset))
          if new_score < old_score*MIN_FRAC_INCR:
            continue
          if new_score < self._min_score:
            continue
          # The new feature set improves the classifier
          # Add the new feature; delete the old ones
          process_dct[new_fset.str] = new_score
          del process_dct[other_fset.str]
          is_changed = True
          break
        if not is_changed:
          update(cur_fset)
      self._ser_comb = pd.Series(result_dct)
      self._ser_comb = self._ser_comb.sort_values(
          ascending=False)
    return self._ser_comb

  def disjointify(self, **kwargs):
    """
    Creates a list of feature set strings with non-overlapping
    features.

    Parameters
    ----------
    kwargs: dict
        Parameters passed to function.

    Returns
    -------
    pd.Series
    """
    return disjointify(self.ser_sbfset, **kwargs)

  def serialize(self, dir_path):
    """
    Serializes the computed objects.

    Parameters
    ----------
    dir_path: str
      Path to the directory where objects are serialized.

    Returns
    -------
    None.
    """
    if not os.path.isdir(dir_path):
      os.mkdir(dir_path)
    # Save the computed values
    for stg in COMPUTES:
      result = eval("self.%s" % stg)
      if result is not None:
        path = os.path.join(dir_path, "%s.csv" % stg)
        result.to_csv(path)
    # Serialize constructor parameter values
    path = os.path.join(dir_path, MISC_PCL)
    persister = Persister(path)
    persister.set(self._min_score)

  @classmethod
  def deserialize(cls, dir_path):
    """
    Deserializes a FeatureSetCollection saved in its
    FeatureAnalyzer directory.

    Parameters
    ----------
    dir_path: str

    Returns
    ----------
    FeatureSetCollection
    """
    def readDF(name):
     # Reads the file for the variable name if it exists
     # Returns a DataFrame or a Series
      result = None
      UNNAMED = "Unnamed: 0"
      path = os.path.join(dir_path, "%s.csv" % name)
      if os.path.isfile(path):
        df = pd.read_csv(path)
        if UNNAMED in df.columns:
          df.index = df[UNNAMED]
          del df[UNNAMED]
          df.index.name = None
        if len(df.columns) == 1:
          result = pd.Series(df[df.columns.tolist()[0]])
      return result
    #
    # Get constructor parameter values
    path = os.path.join(dir_path, MISC_PCL)
    if os.path.isfile(path):
      persister = Persister(path)
      min_score = persister.get()
    else:
      min_score = MIN_SCORE
    # Get the analyzer
    key = "X"
    analyzer_dct = feature_analyzer.deserialize(
        {key: dir_path})
    # Construct the FeatureSetCollection
    collection = cls(analyzer_dct[key], min_score=min_score)
    collection._ser_sbfset = readDF(SER_SBFSET)
    collection._ser_comb = readDF(SER_COMB)
    collection._df_case = readDF(DF_CASE)
    return collection

  def _getNumZero(self, ser_X, fset_selector=lambda f: True):
    """
    Gets the number of zeroes in the significance level
    for feature sets applicable to cases in ser_X.

    Parameters
    ----------
    ser_X: pd.DataFrame
        Feature vector for a single instance
    fset_selector: Function
        Args: fset
        Returns: bool

    Returns
    -------
    list-FeatureSet, list-float
    """
    fset_stgs = self.df_case[cn.FEATURE_SET].unique()
    fsets = [FeatureSet(f) for f 
        in fset_stgs if fset_selector(FeatureSet(f))]
    num_zeroes = []
    for fset in fsets:
      case_in_ser = Case(fset, 
          ser_X.loc[fset.list].values)
      df_fset = self.df_case[
          self.df_case[cn.FEATURE_SET] == fset.str]
      cases = [Case(fset, df_fset.loc[i, cn.CASE])
          for i in df_fset.index]
      sel = [case_in_ser.equals(c) for c in cases]
      num_zero = df_fset[sel][cn.NUM_ZERO].values[0]
      num_zeroes.append(num_zero)
    return fsets, num_zeroes

  def plotEvaluateHistogram(self, ser_X, ax=None,
      title="", xlim=(-11, 11),
      is_plot=True, max_count=None,
      fset_selector=lambda f: True):
    """
    Plots the results of a feature vector evaluation.

    Parameters
    ----------
    ser_X: pd.DataFrame
        Feature vector for a single instance
    is_plot: bool
    fset_selector: Function
        Args: fset
        Returns: bool

    Returns
    -------
    None.
    """
    _, num_zeroes = self._getNumZero(ser_X,
        fset_selector=fset_selector)
    if max_count is None:
      counter = collections.Counter(num_zeroes)
      max_count = counter.most_common(1)[0][1]
    if ax is None:
      fig, ax = plt.subplots()
    ax.hist(num_zeroes)
    ax.set_xlabel("0s in SL")
    ax.set_ylabel("count")
    ax.set_xlim(xlim)
    ax.set_ylim([0, max_count])
    ax.set_title(title)
    ax.plot(xlim, [0, 0], color="black")
    ax.plot([0, 0], [0, max_count], color="black",
        linestyle=":")
    if is_plot:
      plt.show()
      

  def plotEvaluate(self, ser_X, num_fset=3, ax=None,
      title="", ylim=(-5, 5), label_xoffset=-0.2,
      is_plot=True, is_include_neg=True,
      fset_selector=lambda f: True,
      **kwargs):
    """
    Plots the results of a feature vector evaluation.

    Parameters
    ----------
    ser_X: pd.DataFrame
        Feature vector for a single instance
    num_fset: int
        Top feature sets selected
    is_plot: bool
    label_xoffset: int
        How much the text label is offset from the bar
        along the x-axis
    is_include_neg: bool
        Plot neg class as neg values
    fset_selector: Function
        Args: fset
        Returns: bool
    kwargs: dict
        optional arguments for FeatureSet.evaluate

    Returns
    -------
    None.
    """
    def convert(v):
      if v < 0:
        v = np.log10(-v)
      elif v > 0:
        v = -np.log10(v)
      else:
        raise ValueError("Should not be 0.")
      return v
    #
    # Initializations
    df_X = pd.DataFrame(ser_X).T
    fsets = [FeatureSet(s, analyzer=self._analyzer)
        for s in self.ser_comb.index.tolist()[0:num_fset]]
    # Construct labels
    labels = [f.str for f in fsets]
    # Construct data
    values = []
    for idx, fset in enumerate(fsets):
      if not fset_selector(fset):
        continue
      value = fset.evaluate(df_X, 
          is_include_neg=is_include_neg,
          **kwargs).values[0]
      if np.isnan(value):
        raise RuntimeError("Should not get nan")
        #labels[idx] = "*%s" % labels[idx]
        #values.append(min_sl)
      else:
        values.append(fset.evaluate(df_X).values[0])
    # Construct plot Series
    ser_plot = pd.Series(values)
    ser_plot.index = ["" for _ in range(len(labels))]
    ser_plot = pd.Series([convert(v) for v in ser_plot])
    # Bar plot
    width = 0.1
    if ax is None:
      fig, ax = plt.subplots()
      # ax = ser_plot.plot(kind="bar", width=width)
    ax.bar(labels, ser_plot, width=width)
    ax.set_ylabel("0s in SL")
    ax.set_xticklabels(ser_plot.index.tolist(),
        rotation=0)
    ax.set_ylim(ylim)
    ax.set_title(title)
    for idx, label in enumerate(labels):
      if is_include_neg:
        ypos = ylim[0] + 1
      else:
        ypos = 0.25
      xpos = idx + label_xoffset
      ax.text(xpos, ypos, label, rotation=90,
          fontsize=8)
    # Add the 0 line if needed
    if is_include_neg:
      ax.plot([0, len(labels)-0.75], [0, 0],
          color="black")
    if is_plot:
      plt.show()

  def plotProfileInstance(self, fsets, is_plot=True,
      **kwargs):
    """
    Profile plots for feature sets.
    :param list-FeatureSet/list-str
    :param bool is_plot:
    :param dict kwargs: options for plot:
    """
    count = len(fsets)
    fig, axes = plt.subplots(1, count, **kwargs)
    x_spacing = 3*count
    for idx, fset in enumerate(fsets):
      fset = FeatureSet(fset, analyzer=self._analyzer)
      accuracy = self.ser_comb.loc[fset.str]
      title = "Acc: %2.2f" % accuracy
      fset.plotProfileInstance(ax=axes[idx], title=title,
          is_plot=False, x_spacing=x_spacing)
    if is_plot:
      plt.show()

if __name__ == '__main__':
  msg = "Construct FeatureSetCollection metrics."
  parser = argparse.ArgumentParser(description=msg)
  msg = "Absolute path to the FeatureAnalyzer"
  msg += " serialization directory. Also where"
  msg += " results are stored."
  parser.add_argument("path", help=msg, type=str)
  args = parser.parse_args()
  key = "X"
  if False:
    analyzer_dct = feature_analyzer.deserialize(
        {key: args.path})
    collection = FeatureSetCollection(analyzer_dct[key])
  collection = FeatureSetCollection.deserialize(args.path)
  collection.serialize(args.path)
