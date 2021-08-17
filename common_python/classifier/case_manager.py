'''Constructs cases for classification.'''

"""
A Case is a statistically significant FeatureVector in that it distinguishes
between classes.
Herein, only binary classes are considered (0, 1).
A CaseManager constructs cases, provides an efficient way to search
cases based on a feature vector, and reports the results.

The significance level of a case is calculaed based on the frequency of
case occurrence for labels using a binomial null distribution with p=0.5.
"""

from common_python.statistics.binomial_distribution import BinomialDistribution
from common_python.classifier.feature_set import FeatureVector

import collections
import matplotlib.pyplot as plt
import numpy as np
import sklearn


MAX_SL = 0.05
# Random forest defaults
RANDOM_FOREST_DEFAULT_DCT = {
    "max_depth": 4,
    "random_state": 0,
    "bootstrap": False,
    "min_impurity_decrease": 0.01,
    "min_samples_leaf": 5,
    }

FeatureVectorStatistic = collections.namedtuple("FeatureVectorStatistic",
    "sl cnt pos")


class Case:
  """Case for a binary classification."""

  def __init__(self, feature_vector, fv_statistic, dtree=None):
    """
    Parameters
    ----------
    feature_vector: FeatureVector
    fv_statistic: FeatureVectorStatistic
    dtree: sklearn.DecisionTreeClassifier
        Tree from which case was constructed
    """
    self.feature_vector = feature_vector
    self.fv_statistic = fv_statistic
    self.dtree = dtree

  def __repr__(self):
    return "%s: %3.2f" % (str(self.feature_vector), self.fv_statistic.sl)

  def displayTree(self, feature_names, is_display=True,
      figsize=(12, 10), fontsize=18):
    """
    Creates a graph of the decision tree used to construct the Case.
    Tree tests are the left branch; False is the right branch.

    Parameters
    ----------
    feature_names: list-str
        names of all features used in classification
    """
    # Creates plots in matplotlib
    _ = plt.figure(figsize=figsize)
    # the clf is Decision Tree object
    if is_display:
      _ = sklearn.tree.plot_tree(self.dtree, feature_names=feature_names,
          fontsize=fontsize, filled=True)


class CaseManager:

  def __init__(self, df_X, ser_y, max_sl=MAX_SL):
    """
    Parameters
    ----------
    df_X: pd.DataFrame
        columns: features
        index: instance
        value: feature value
    ser_y: pd.Series
        index: instance
        value: class {0, 1}
    max_sl: float
        maximum significance level for a case
    binomial_prob: float
        prior probability of class 1
    """
    self._df_X = df_X
    self._ser_y = ser_y
    self._max_sl = max_sl
    self._forest = None  # Random forest used to construct cases
    self._df_case = None  # Dataframe representation of cases
    self._features = list(df_X.columns)
    self._min_num_sample = np.log10(self._max_sl)/np.log10(binomial_prob)
    self.cases = None  # Cases constructed during by calling build
    total_sample = len(self._ser_y)
    prior_prob = sum(self._ser_y) / total_sample
    self._binom = BinomialDistribution(total_sample, prior_prob)

  def _getCompatibleFeatureValues(self, feature, value_ub=None):
    """
    Finds the values of the feature that are compatible in the sense that
    they are less than or equal to the upper bound.

    Parameters
    ----------
    feature: str
    value_ub: float
        None: return all values

    Returns
    -------
    list-float
    """
    values = self._df_X.loc[:, feature]
    value_sub = list(set([v for v in values if v <= value_ub]))
    value_sub.sort()
    return value_sub

  def _getIndices(self, feature_vector):
    """
    Finds the indices selected by the feature vector.

    Parameters
    ----------
    feature_vector: FeatureVector
    
    Returns
    -------
    list-object
    """
    indices = set(self._df_X.index)
    for feature, value in feature_vector.dict.items():
      these_indices = self._df_X[df_X[feature] == value].index
      indices = indices.intersection(these_indices)
    return list(indices)

  def _getFeatureVectorStatistic(self, feature_vector):
    """
    Calculates the significance level of the feature vector and other statistics
    for the feature and classification data.

    Parameters
    ----------
    feature_vector: FeatureVector
    
    Returns
    -------
    FeatureVectorStatistic
    """
    indices = self._getIndices(feature_vector)
    num_sample = len(indices)
    num_pos = sum(self._ser_y.loc[indices])
    return FeatureVectorStatistic(
        sl=self._binom.getSL(num_sample, num_pos),
        cnt=num_sample,
        pos=num_pos,
        )

  def _getCases(self, dtree):
    """
    Finds the feature vectors in the tree that meet the significance level
    requirement.
    1. Once a feature vector is significant, no extension of it is explored.
    2. Feature vectors must not exceed a maximum significance level.

    Parameters
    ----------
    dtree: DecisionTreeClassifier

    Returns
    -------
    list-FeatureVector
    """
    def processTree(node, feature_vector=None):
      """
      Recursively constructs cases from feature vectors.

      Usage
      -----
      feature_vectors = processTree(0)

      Parameters
      ----------
      node: int
          node in the decision tree
      feature_vector: FeatureVector

      Returns
      -------
      list-Case
      """
      def processBranch(feature_name, feature_values, feature_dct, branch_nodes):
        """
        Processes a branch in the tree for all possible feature values.
   
        Parameters
        ----------
        feature_name: str
        feature_values: list-float
        branch_nodes: np.array
        
        Returns
        -------
        list-Case
        """
        new_cases = []
        for value in feature_values:
          # Consruct a new FeatureVector
          dct = dict(feature_dct)
          dct[feature_name] = value
          new_feature_vector = FeatureVector(dct)
          # Determine if it is a Case
          fv_statistic = self._getFeatureVectorStatistic(new_feature_vector)
          if fv_statistic.cnt < self._min_num_sample:
            continue
          if np.abs(fv_statistic.sl) < self._max_sl:
            # Statistically significant FeatureVector is a Case.
            new_cases.append(Case(new_feature_vector, fv_statistic,
                dtree=dtree))
          else:
            # feature_vector is not a Case, but maybe an extended vector will be
            new_cases.extend(processTree(branch_nodes[node],
                feature_vector=new_feature_vector))
        return new_cases
      #
      # Check for termination of this recursion
      if dtree.tree_.feature[node] == _dtree.TREE_UNDEFINED:
        return []
      # Initialize this recursion
      feature_dct = {}
      if feature_vector is not None:
        feature_dct = feature_vector.dict
      # Process the node
      feature_name = self._features[dtree.tree_.feature[node]]
      threshold = dtree.tree_.threshold_[node]
      feature_values_all = [self._getCompatibleFeatureValues(feature_name,
          value_ub=None)]
      feature_values_left = [self._getCompatibleFeatureValues(feature_name,
          value_ub=threshold)]
      feature_values_right = list(set(feature_values_all).difference(
          feature_values_left))
      import pdb; pdb.set_trace()
      # Process each branch
      cases = processBranch(feature_name, feature_values_left,
          feature_dct, dtree.tree_.children_left)
      cases.append(processBranch(feature_name, feature_values_right,
          feature_dct, dtree.tree_.children_right))
      #
      return cases
    # Calculate the feature vectors beginning at the root
    return processTree(0)

  def build(self, **kwargs):
    """
    Builds the cases and related internal data.

    Parameters
    ----------
    kwargs: dict
        optional arguments used in random forest to find cases.
    """
    # Create arguments for random forest and run it
    forest_kwargs = dict(kwargs)
    for key, value in RANDOM_FOREST_DEFAULT_DCT.items():
      if not key in forest_kwargs.keys():
        forest_kwargs[key] = value
    self._forest = sklearn.ensemble.RandomForestClassifier(**forest_kwargs)
    self.cases = [self._getCases(t) for t in self._forest.estimators_]
