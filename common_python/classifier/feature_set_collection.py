'''Constructs and evaluates alternative classifier features.'''

import common_python.constants as cn
from common_python.classifier.feature_set  \
    import FeatureSet
from common_python.classifier import util_classifier
from common_python.classifier import feature_analyzer
from common_python.util.persister import Persister

import argparse
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
COMPUTES = [SER_SBFSET, SER_COMB]
MISC_PCL = "feature_set_collection_misc.pcl"

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
      else:
        result = None
      return result
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
    return collection

  def plotProfileFsets(self, fsets, is_plot=True,
      **kwargs):
    """
    Profile plots for feature sets.
    :param list-FeatureSet/list-str
    :param bool is_plot:
    :param dict options for plot:
    """
    count = len(fsets)
    fig, axes = plt.subplots(1, count, **kwargs)
    x_spacing = 3*count
    for idx, fset in enumerate(fsets):
      self.plotProfileFset(fset, ax=axes[idx],
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
