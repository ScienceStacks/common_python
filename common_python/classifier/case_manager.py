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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier


MAX_SL = 0.05
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


##################################################################
class FeatureVectorStatistic:

  def __init__(self, num_sample, num_pos, siglvl):
    self.siglvl = siglvl
    self.num_sample = num_sample
    self.num_pos = num_pos

  def __repr__(self):
    return "smp=%d, pos=%d, sl=%2.2f" % (self.num_sample,
        self.num_pos, self.siglvl)


##################################################################
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
    return "%s: %s" % (str(self.feature_vector), str(self.fv_statistic))


##################################################################
class CaseManager:

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
    self._df_X = df_X
    self._ser_y = ser_y
    self._max_sl = max_sl
    self._forest_kwargs = kwargs
    self._forest = None  # Random forest used to construct cases
    self._df_case = None  # Dataframe representation of cases
    self._features = list(df_X.columns)
    self.case_dct = None  # key: str(FeatureSet), value: Case; sorted order
    total_sample = len(self._ser_y)
    self._prior_prob = sum(self._ser_y) / total_sample
    self._binom = BinomialDistribution(total_sample, self._prior_prob)
    self._min_num_sample = np.log10(self._max_sl)/np.log10(self._prior_prob)

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
    return FeatureVectorStatistic(num_sample, num_pos,
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
    dict
        key: str(FetureVector)
        value: case
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
    return {str(c.feature_vector): c for c in cases}

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
      cases = list(self.case_dct.values())
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
    self.case_dct = {}
    for dtree in self._forest.estimators_:
        new_case_dct = self._getCases(dtree)
        for key, value in new_case_dct.items():
          self.case_dct[key] = value
    # Sort the cases
    stgs = list(self.case_dct.keys())
    stgs.sort()
    self.case_dct = {k: self.case_dct[k] for k in stgs}

  def plotEvaluate(self, ser_X, max_sl=MAX_SL, ax=None,
      title="", ylim=(-5, 5), label_xoffset=-0.2,
      is_plot=True):
    """
    Plots the results of a feature vector evaluation.

    Parameters
    ----------
    ser_X: pd.DataFrame
        Feature vector to be evaluated
    max_sl: float
        Maximum significance level to plot
    ax: Axis for plot
    is_plot: bool
    label_xoffset: int
        How much the text label is offset from the bar
        along the x-axis

    Returns
    -------
    list-case
        cases plotted
    """
    if self.case_dct is None:
      raise ValueError("Must do `build` first!")
    MIN_SL = 1e-5
    #
    def convert(v):
      if v < 0:
        v = max(v, -MIN_SL)
        v = np.log10(-v)
      elif v > 0:
        v = max(v, MIN_SL)
        v = -np.log10(v)
      else:
        raise ValueError("Should not be 0.")
      return v
    #
    feature_vector = FeatureVector(ser_X.to_dict())
    cases = [c for c in self.case_dct.values()
         if feature_vector.isCompatible(c.feature_vector)
         and np.abs(c.fv_statistic.siglvl) < max_sl]
    # Select applicable cases
    feature_vectors = [c.feature_vector for c in cases]
    siglvls = [c.fv_statistic.siglvl for c in cases]
    count = len(cases)
    # Construct plot Series
    # Do the plot
    if is_plot and (count == 0):
        print("***No Case found for %s" % title)
    else:
      ser_plot = pd.Series(siglvls)
      ser_plot.index = ["" for _ in range(count)]
      labels  = [str(c) for c in feature_vectors]
      ser_plot = pd.Series([convert(v) for v in ser_plot])
      # Bar plot
      width = 0.1
      if ax is None:
        _, ax = plt.subplots()
        # ax = ser_plot.plot(kind="bar", width=width)
      ax.bar(labels, ser_plot, width=width)
      ax.set_ylabel("0s in SL")
      ax.set_xticklabels(ser_plot.index.tolist(),
          rotation=0)
      ax.set_ylim(ylim)
      ax.set_title(title)
      for idx, label in enumerate(labels):
        ypos = ylim[0] + 1
        xpos = idx + label_xoffset
        ax.text(xpos, ypos, label, rotation=90,
            fontsize=8)
      # Add the 0 line if needed
      ax.plot([0, len(labels)-0.75], [0, 0],
          color="black")
      ax.set_xticklabels([])
    if is_plot:
      plt.show()
    return cases

  @classmethod
  def mkCaseManagers(cls, df_X, ser_y, **kwargs):
    """
    Constructs a CaseManager for each class in ser_y and builds the cases.

    Parameters
    ----------
    df_X: pd.DataFrame
    ser_y: pd.Series
    dict: keyword parameters for CaseManager

    Returns
    -------
    dict
        key: class
        value: CaseManager
    """
    manager_dct = {}
    classes = list(set(ser_y.values))
    for a_class in classes:
      new_ser_y = ser_y.apply(lambda v: 1 if v == a_class else 0)
      manager_dct[a_class] = cls(df_X, new_ser_y, **kwargs)
      manager_dct[a_class].build()
    return manager_dct

  def filterCaseByDescription(self, ser_desc,
      include_terms=None, exclude_terms=None):
    """
    Use "or" semantics to select cases pruning those not selected.
    A case is selected if at least one of the terms in include_terms is
    in the description for the feature or if at least one of the terms in
    exclude_terms is *not* in the feature description.
    "And" semantics are achieved by calling this method multiple times.

    Parameters
    ----------
    ser_desc: pd.Series
        key: feature
        value: str
    include_terms: list-str
        terms, one of which must be present
    exclude_terms: list-str   
        terms, one of which must be absent

    State
    -----
    self.case_dct: Updated
    """
    # Initializations
    common_features = set(self._features).intersection(ser_desc.index)
    ser_desc_sub = ser_desc.loc[common_features]
    #
    def findFeaturesWithTerms(terms):
      """
      Finds features with descriptions that contain at least one term.  

      Parameters
      ----------
      terms: list-str

      Returns
      -------
      list-features
      """
      sel = ser_desc.copy()
      sel = sel.apply(lambda v: False)
      for term in terms:
          sel = sel |  ser_desc_sub.str.contains(term)
      return list(ser_desc_sub[sel].index)
    #
    def findCasesWithFeature(features):
      """
      Finds the cases that have at least one of the features.
 
      Parameters
      ----------
      features: list-Feature
      
      Returns
      -------
      list-Case
      """
      cases = []
      for feature in features:
        cases.extend([c for c in self.case_dct.values()
            if feature in c.feature_vector.fset.list])
      return list(set(cases))
    #
    # Include terms
    if include_terms is None:
      selected_cases = []
    else:
      include_features = findFeaturesWithTerms(include_terms)
      selected_cases = findCasesWithFeature(include_features)
    # Exclude terms
    if exclude_terms is None:
      satisfy_exclude_cases = set([])
    else:
      exclude_features = findFeaturesWithTerms(exclude_terms)
      term_absent_cases = findCasesWithFeature(exclude_features)
      satisfy_exclude_cases = set(self.case_dct.values()).difference(
          term_absent_cases)
    #
    accepted_cases = list(satisfy_exclude_cases.union(selected_cases))
    self.case_dct = {str(c.feature_vector): c for c in accepted_cases}
