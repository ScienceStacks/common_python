'''Constructs and evaluates alternative classifier features.'''

import common_python.constants as cn
from common_python.classifier.feature_set  \
    import FeatureSet, FeatureVector
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
import typing

MIN_FRAC_INCR = 1.01  # Must increase by at least 1%
MIN_SCORE = 0
SER_SBFSET = "ser_sbfset"
SER_COMB = "ser_comb"
DF_FV = "df_fv"
COMPUTES = [SER_SBFSET, SER_COMB, DF_FV]
MISC_PCL = "feature_set_collection_misc.pcl"
MIN_SL = 1e-10
MAX_SL = 1

############ FUNCTIONS #################
def disjointify(ser_fset, min_score=MIN_SCORE):
  """
  Makes the feature sets disjoint by discarding
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
    self._df_fv = None  # Feature vectors in cases

  @property
  def df_fv(self):
    """
    Dataframe describing feature vectors.
    :return pd.Series:
        Columns: cn.FEATURE_SET, cn.FEATURE_VECTOR, cn.SIGLVL_ZEROES
    Notes:
      1. cn.SIGLVL_ZEROES may be negative to 
         indicate significance level for a negative case.
    """
    return self._makeFeatureVector()

  def _makeFeatureVector(self): 
    """
    Dataframe describing the FeatureVectors.
    :return pd.Series:
        Columns: 
          cn.FEATURE_SET
          cn.FEATURE_VECTOR - string representation of case
          cn.NUM_ZERO - number of zeroes in the
                        significance level; positive
                        for positive case and negative
                        for negative case.
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
    if self._df_fv is None:
      dct = {
          cn.FEATURE_SET: [], cn.FEATURE_VECTOR: [],
          cn.NUM_ZERO: [] }
      for fset_stg in self.ser_comb.index:
        fset = FeatureSet(fset_stg,
            analyzer=self._analyzer)
        df_profile = fset.profileTrinary()
        for feature_tuple in df_profile.index:
          siglvl_pos = df_profile.loc[[feature_tuple],
               cn.SIGLVL_POS][0]
          siglvl_neg = df_profile.loc[[feature_tuple],
               cn.SIGLVL_NEG][0]
          num_zero = convert(siglvl_pos, siglvl_neg)
          dct[cn.FEATURE_SET].append(fset.str)
          dct[cn.FEATURE_VECTOR].append(feature_tuple)
          dct[cn.NUM_ZERO].append(num_zero)
      self._df_fv = pd.DataFrame(dct)
    return self._df_fv
        
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
    if self._ser_comb is None:
      self._ser_comb = _mkSerComb()
    return self._ser_comb

  def _mkSerComb(self, max_size=None, max_search=None):
    """
    Creates feature sets up to a maximum size.

    Parameters
    ----------
    max_size: int
        Maximum size of the feature set collection
    max_search: int
        Maximum number of feature sets searched

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
    def getScore(fset):
      # Gets the score for an fset
      return process_dct[fset.str]
    #
    if self._ser_comb is not None:
      ser_comb = self._ser_comb.copy()
    else:
      ser_comb = pd.Series(dtype=float)
    #
    if max_size is not None:
      if len(ser_comb) >= max_size:
        return ser_comb
    ser = self.disjointify(min_score=self._min_score)
    process_dct = ser.to_dict()
    result_dct = {}
    #
    # Iteratively consider combinations of fsets
    while len(process_dct) > 0:
      if len(result_dct) >= max_size:
        break
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
      if max_search is None:
          max_search = len(process_dct)
      for idx in range(max_search):
        other_fset_stg = list(process_dct.keys())[idx]
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
    #
    ser_comb = pd.Series(result_dct)
    if len(ser_comb) < 0:
      ser_comb = self._ser_comb.sort_values(ascending=False)
    return ser_comb

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
      df = None
      UNNAMED = "Unnamed: 0"
      path = os.path.join(dir_path, "%s.csv" % name)
      if os.path.isfile(path):
        df = pd.read_csv(path)
        if UNNAMED in df.columns:
          df.index = df[UNNAMED]
          del df[UNNAMED]
          df.index.name = None
        if len(df.columns) == 1:
          df = pd.Series(df[df.columns.tolist()[0]])
      return df
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
    collection._df_fv = readDF(DF_FV)
    # FIXME: Corrected for change in name
    collection._df_fv = collection._df_fv.rename(
        columns={"case": cn.FEATURE_VECTOR})
    collection._df_fv[cn.FEATURE_VECTOR] = [eval(v) for v in
      collection._df_fv[cn.FEATURE_VECTOR]]
    return collection

  def _getNumZero(self, ser_X,
      fset_selector=lambda f: True, max_sl=1):
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
    min_num_zero = np.abs(np.log10(max_sl))
    fset_stgs = self.df_fv[cn.FEATURE_SET].unique()
    fsets = [FeatureSet(f) for f 
        in fset_stgs if fset_selector(FeatureSet(f))]
    num_zeroes = []
    for fset in fsets:
      vector_in_ser = fset.getFeatureVector(ser_X)
      df_fset = self.df_fv[
          self.df_fv[cn.FEATURE_SET] == fset.str]
      sel = [vector_in_ser.tuple == t for t in
          df_fset[cn.FEATURE_VECTOR]]
      num_zero = df_fset[sel][cn.NUM_ZERO].values
      if len(num_zero) == 1:
        num_zeroes.append(num_zero[0])
      else:
        print("Missing feature vector for %s" % fset.str)
    result = [(f, n) for f, n in zip(fsets, num_zeroes)
        if np.abs(n) >= min_num_zero]
    return [list(r) for r in zip(*result)]

  def plotEvaluateHistogram(self, ser_X, ax=None,
      title="", xlim=(-11, 11), ylim=None,
      is_plot=True, max_count=None, max_sl=1,
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
    max_sl: float
        Maximum significance level plotted
    max_count: int
        Maximum count of occurrences

    Returns
    -------
    None.
    """
    result = self._getNumZero(ser_X,
        fset_selector=fset_selector, max_sl=max_sl)
    if len(result) == 0:
      return
    _, num_zeroes = result
    if max_count is None:
      counter = collections.Counter(num_zeroes)
      max_count = counter.most_common(1)[0][1]
    if ax is None:
      fig, ax = plt.subplots()
    ax.hist(num_zeroes, bins=xlim[1]-xlim[0]+1)
    ax.set_xlabel("0s in SL")
    ax.set_ylabel("count")
    ax.set_xlim(xlim)
    if ylim is None:
      ylim = [0, max_count]
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.plot(xlim, [0, 0], color="black")
    ax.plot([0, 0], [0, ylim[1]], color="black",
        linestyle=":")
    if is_plot:
      plt.show()

  def fullEvaluate(ser_X, title="", **kwargs):
    """
    Plots all evaluate profiles.

    Parameters
    ----------
    ser_X: pd.DataFrame
        Feature vector for a single instance
    kwargs: dict
        optional arguments for FeatureSet.evaluate

    Returns
    -------
    None.
    """
    num_row = 2 
    num_col = 3 
    fig, axes = plt.subplots(num_row, num_col,
      figsize=(16, 10))
    for idx, state in enumerate(STATES):
      row = int(idx/num_col)
      col = idx % num_col
      collection = COLLECTION_DCT[state]
      if row == 0:
        label_xoffset = -0.1
      else:
        label_xoffset = 0.1 
      collection.plotEvaluate(ser_X, 
        ax=axes[row, col], is_plot=False,
        title = "State %d" % idx,
        label_xoffset=label_xoffset, **kwargs)
    fig.suptitle(title, fontsize=16)
    plt.show()
      
  def getFVEvaluations(self, ser_X:pd.Series, num_fset:int=3,
      max_sl:float=0.01,
      fset_selector:typing.Callable=lambda f: True,
      is_include_neg:bool=True, **kwargs):
    """
    Constructs metrics for a feature vector evaluation.

    Parameters
    ----------
    ser_X: Feature vector for a single instance
    num_fset: Top feature sets selected
    max_sl: maximum significance level considered
    is_include_neg: include neg class as neg values
    fset_selector: Function
        Args: fset
        Returns: bool
    kwargs: dict
        optional arguments for FeatureSet.evaluate

    Returns
    -------
    pd.Dataframe
       Columns from FeatureSet.evaluate
       FEATURE_SET
    """
    # Initializations
    df_X = pd.DataFrame(ser_X).T
    num = min(num_fset, len(self.ser_comb))
    fsets = [FeatureSet(s, analyzer=self._analyzer)
        for s in self.ser_comb.index.tolist()[0:num]]
    # Construct data
    siglvls = []
    feature_vectors = []
    dfs = []
    for fset in fsets:
      df = fset.evaluate(df_X, 
          is_include_neg=is_include_neg, **kwargs)
      df_sub = df[df[cn.SIGLVL] < max_sl]
      df_sub = df_sub[df_sub[cn.COUNT] >= num_fset]
      df_sub[cn.FEATURE_SET] = fset
      dfs.append(df_sub)
    df_result = pd.concat(dfs)
    return df_result
      
  def plotEvaluate(self, ser_X, is_include_neg=True, ax=None,
      title="", ylim=(-5, 5), label_xoffset=-0.2,
      is_plot=True, **kwargs):
    """
    Plots the results of a feature vector evaluation.

    Parameters
    ----------
    ser_X: pd.DataFrame
        Feature vector to be evaluated
    ax: Axis for plot
    is_include_neg: bool
        Include negative cases
    is_plot: bool
    label_xoffset: int
        How much the text label is offset from the bar
        along the x-axis
    kwargs: dict
        optional arguments for constructing evaluation data

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
    df = self.getFVEvaluations(ser_X,
        is_include_neg=is_include_neg, **kwargs)
    feature_vectors = df[cn.FEATURE_VECTOR]
    siglvls = df[cn.SIGLVL]
    count = len(feature_vectors)
    # Construct plot Series
    if count == 0:
      print("***No fset found that is consistent with the feature vector.")
    else:
      ser_plot = pd.Series(siglvls)
      ser_plot.index = ["" for _ in range(count)]
      labels  = [str(c) for c in feature_vectors]
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
      ax.set_xticklabels([])
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
