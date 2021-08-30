'''Builds cases for classification.'''

"""
A CaseBuilder constructs cases.

The significance level of a case is calculaed based on the frequency of
case occurrence for labels using a binomial null distribution with p=0.5.

TODO:
1. Remove contradictory cases (positive SL on > 1 class)
2. Handle correlated cases
"""

from common_python.classifier.cc_case.case_core import  \
    Case, FeatureVectorStatistic
from common_python.classifier.cc_case.case_collection import CaseCollection
from common_python.classifier.cc_case.case_multi_collection  \
    import CaseMultiCollection
from common_python.statistics.binomial_distribution import BinomialDistribution
from common_python.classifier.feature_set import FeatureVector

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier


MAX_SL = 0.05
MIN_SL = 1e-5  # Minimum value of significance level used
               # in conerting to nuumber of zeroes
# Random forest defaults
RANDOM_FOREST_DEFAULT_DCT = {
    "n_estimators": 200,  # Number of trees created
    "max_depth": 4,
    "random_state": 0,
    "bootstrap": False,
    "min_impurity_decrease": 0.01,
    "min_samples_leaf": 5,
    }
TREE_UNDEFINED = -2
IS_CHECK = True  # Does additional tests of consistency


##################################################################
class CaseBuilder:

  def __init__(self, df_X, ser_y, max_sl=MAX_SL, **kwargs):
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
    kwargs: dict
        optional arguments used in random forest to find cases.
    """
    self.version = 2 # Change version when internal state changes
    self._df_X = df_X
    self._ser_y = ser_y
    self._max_sl = max_sl
    self._forest_kwargs = kwargs
    self._forest = None  # Random forest used to construct cases
    self._df_case = None  # Dataframe representation of cases
    self._features = list(df_X.columns)
    self.case_col = CaseCollection({}, df_X=self._df_X, ser_y=self._ser_y)
    total_sample = len(self._ser_y)
    self._prior_prob = sum(self._ser_y) / total_sample
    self._binom = BinomialDistribution(total_sample, self._prior_prob)
    self._min_num_sample = np.log10(self._max_sl)/np.log10(self._prior_prob)

  def _getCompatibleFeatureValues(self, feature, value_ub=None):
    """
    Finds the values of the feature that are compatible in the sense that
    they are less than or equal to the upper bound. This is part of the
    analysis of the DecisionTree nodes.

    Parameters
    ----------
    feature: str
    value_ub: float
        None: return all values

    Returns
    -------
    list-float
    """
    values = self._df_X.loc[:, feature].to_list()
    if value_ub is None:
      value_sub = values
    else:
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
      these_indices = self._df_X[self._df_X[feature] == value].index
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
    return FeatureVectorStatistic(num_sample, num_pos, self._prior_prob,
        self._binom.getSL(num_sample, num_pos))

  def _getCases(self, dtree):
    """
    Finds the feature vectors in the tree that meet the significance level
    requirement.
    Feature vectors must not exceed a maximum significance level.

    Parameters
    ----------
    dtree: DecisionTreeClassifier

    Returns
    -------
    CaseCollection
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
      dict
          key: str(FetureVector)
          value: case
      """
      def processBranch(feature_name, feature_values, feature_dct, branch_nodes):
        """
        Processes a branch in the tree for all possible feature values.

        Parameters
        ----------
        feature_name: str
            child feature to process
        feature_values: list-float
            child values to use
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
          if fv_statistic.num_sample < self._min_num_sample:
            continue
          if np.abs(fv_statistic.siglvl) < self._max_sl:
            # Statistically significant FeatureVector is a Case.
            new_cases.append(Case(new_feature_vector, fv_statistic,
                dtree=dtree))
          child = branch_nodes[node]
          new_cases.extend(processTree(child, feature_vector=new_feature_vector))
        return new_cases
      #
      # Check for termination of this recursion
      if dtree.tree_.feature[node] == TREE_UNDEFINED:
        return []
      # Initialize this recursion
      feature_dct = {}
      if feature_vector is not None:
        feature_dct = feature_vector.dict
      # Process the node
      feature_name = self._features[dtree.tree_.feature[node]]
      threshold = dtree.tree_.threshold[node]
      feature_values_all = list(set(self._getCompatibleFeatureValues(
          feature_name, value_ub=None)))
      feature_values_left = list(set(self._getCompatibleFeatureValues(
          feature_name, value_ub=threshold)))
      feature_values_right = list(set(feature_values_all).difference(
          feature_values_left))
      # Process each branch
      cases = processBranch(feature_name, feature_values_left,
          feature_dct, dtree.tree_.children_left)
      right_cases = processBranch(feature_name, feature_values_right,
          feature_dct, dtree.tree_.children_right)
      cases.extend(right_cases)
      #
      return cases
    # Calculate the feature vectors beginning at the root
    cases = processTree(0)
    return CaseCollection.make(cases)

  def displayCases(self, cases=None, is_display=True,
      figsize=(12, 10), fontsize=14):
    """
    Creates a graph of the decision tree used to construct the Case.
    Tree tests are the left branch; False is the right branch.

    Parameters
    ----------
    feature_names: list-str
        names of all features used in classification
    """
    if cases is None:
      cases = list(self.case_col.values())
    for case in cases:
      # Creates plots in matplotlib
      _, ax = plt.subplots(1, figsize=figsize)
      # the clf is Decision Tree object
      if is_display:
        _ = sklearn.tree.plot_tree(case.dtree, feature_names=self._df_X.columns,
            fontsize=fontsize, filled=True, ax=ax)
        ax.set_title(str(case), fontsize=fontsize)

  def build(self):
    """
    Builds the cases and related internal data.
    """
    # Create arguments for random forest and run it
    forest_kwargs = dict(self._forest_kwargs)
    for key, value in RANDOM_FOREST_DEFAULT_DCT.items():
      if not key in forest_kwargs.keys():
        forest_kwargs[key] = value
    self._forest = RandomForestClassifier(**forest_kwargs)
    self._forest.fit(self._df_X, self._ser_y)
    # Aggregate cases across all decision trees
    self.case_col = CaseCollection({}, df_X=self._df_X, ser_y=self._ser_y)
    for dtree in self._forest.estimators_:
        new_case_col = self._getCases(dtree)
        self.case_col.update(new_case_col)
    # Sort the cases
    self.case_col.sort()

  @classmethod
  def make(cls, df_X, ser_y, **kwargs):
    """
    Constructs a CaseMultiCollection from the data.

    Parameters
    ----------
    df_X: pd.DataFrame
    ser_y: pd.Series
    names: list-str
        names to use for class
    kwargs: dict
        optional arguments for CaseBuilder constructor

    Returns
    -------
    dict
        key: class
        value: CaseBuilder
    """
    dct = {}
    classes = list(set(ser_y.values))
    for a_class in classes:
      new_ser_y = ser_y.apply(lambda v: 1 if v == a_class else 0)
      builder = cls(df_X, new_ser_y, **kwargs)
      builder.build()
      dct[a_class] = builder.case_col
    return CaseMultiCollection(dct)
