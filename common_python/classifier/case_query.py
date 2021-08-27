"""
A CaseQuery selects and presents Cases for a single class.
CaseQuery has static methods that begin with "select" that take
the same inputs and output a dictionary of cases.

A CaseCollectionQuery selects and presents Cases for multiple classes.
CaseCollectionQuery uses the static query methods in CaseQuery.

TODO:
1. selectCaseByDescription - operates on all classes
2. selectCaseByFeatureVector - operates on all classes
   CaseManager returns cases; allows for optional set of cases to select from
"""

from common_python.classifier.case_manager import CaseManager, CaseCollection
from common_python.util import util
import common_python.constants as cn

import copy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


################## FUNCTIONS ##############################
def checkKwargs(keyword_values):
  """
  Verifies the presence of keywords.

  Parameters
  ----------
  keyword_values: list-object
  """
  if any([v is None for v in keyword_values]):
    raise RuntimeError("A keyword %s is unexpectedly None" % " ".join(keywords))


################## CLASSES ################################
class CaseQuery:
  """Selects Cases for a single class and displays results."""

  def __init__(self, case_col):
    """
    Parameters
    ----------
    case_col: CaseCollection
      
    """
    self.case_col = CaseCollection(case_col)

  def __len__(self):
    return len(self.case_col)

  def __eq__(self, other_query):
    return self.case_col == other_query.case_col

  @classmethod
  def select(cls, method, case_query, **kwargs):
    """
    Performs the requested query and returns a CaseQuery.

    Parameters
    ----------
    method: Method
        first argument: CaseCollection
        returns sorted CaseCollection
    case_query: CaseQuery
    kwargs: dict
        keyword arguments for method
    
    Returns
    -------
    CaseQuery
    """
    return cls(method(case_query.case_col, **kwargs))

  ######### STATIC METHOD QUERIES ###############
  # All are static methods that return a CaseCollection
  @staticmethod
  def union(case_col, other_query=None):
    """
    Constructs the union of CaseCollection

    Parameters
    ----------
    case_col: CaseCollection
    other_query: CaseQuery
    
    Returns
    -------
    CaseCollection
    """
    checkKwargs([other_query])
    return case_col.union(other_query.case_col)

  @staticmethod
  def intersection(case_col, other_query=None):
    """
    Constructs the intersection of CaseCollection

    Parameters
    ----------
    case_col: CaseCollection
    other_query: CaseQuery
    
    Returns
    -------
    CaseCollection
    """
    checkKwargs([other_query])
    return case_col.intersection(other_query.case_col)

  @staticmethod
  def difference(case_col, other_query=None):
    """
    Constructs the difference of two CaseCollection.

    Parameters
    ----------
    case_col: CaseCollection
    other_query: CaseQuery
    
    Returns
    -------
    CaseCollection
    """
    checkKwargs([other_query])
    return case_col.difference(other_query.case_col)

  @staticmethod
  def selectByDescription(case_col, ser_desc=None, terms=None):
    """
    Returns those cases that have at least one of the terms.

    Parameters
    ----------
    case_col: CaseCollection
    ser_desc: pd.Series
        key: feature
        value: str
    terms: list-str
        if None, return all cases

    Returns
    -------
    CaseCollection
    """
    checkKwargs([ser_desc])
    if terms is None:
      return [c for c in self.case_col.values()]
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
      sel = ser_desc.apply(lambda v: False)
      for term in terms:
          sel = sel |  ser_desc.str.contains(term)
      return list(ser_desc[sel].index)
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
      selected_cases = []
      for feature in features:
        selected_cases.extend([c for c in case_col.values()
            if feature in c.feature_vector.fset.list])
      return selected_cases
    #
    features = findFeaturesWithTerms(terms)
    return CaseCollection.make(findCasesWithFeature(features))

  @staticmethod
  def selectByFeatureVector(case_col, feature_vector=None):
    """
    Updates self.case_col so that cases must contain the sub-vector
    include_fv and cannot have the sub-vector exclude_fv.
    If they are None, the test is ignored.

    Parameters
    ----------
    case_col: CaseCollection
    feature_vector: FeatureVector

    Returns
    -----
    dict
    """
    checkKwargs([feature_vector])
    cases = [c for c in case_col.values()
        if c.feature_vector.contains(feature_vector)]
    return CaseCollection.make(cases)


###########################################################
class CaseCollectionQuery:
  """Selects subsets of cases and displays results."""

  def __init__(self, dct):
    """
    Parameters
    ----------
    dct: dict
        key: collection identifier (e.g., class name)
        value: CaseCollection / CaseQueryCollection
    """
    self.names = list(dct.keys())
    if isinstance(dct[self.names[0]], CaseQuery):
      self.case_query_dct = dct
    else:
      self.case_query_dct = {k: CaseQuery(v) for k, v in dct.items()}
    
  def toDataframe(self):
    """
    Creates a dataframe of significance levels by class.

    Returns
    -------
    pd.DataFrame
        index: feature_vector
        column: class
        value: significance level
    """
    sers = [c.case_col.toDataframe()[cn.SIGLVL] for c in 
        self.case_query_dct.values()]
    df = pd.concat(sers, axis=1)
    df.columns = self.names
    return df

  def select(self, method, **kwargs):
    """
    Applies the query to each CaseCollection.

    Parameters
    ----------
    method: method that takes CaseCollection as first argument
    
    Returns
    -------
    CaseCollectionQuery
    """
    dct = {}
    for name, case_query in self.case_query_dct.items():
      dct[name] = CaseQuery.select(method, case_query, **kwargs)
    return CaseCollectionQuery(dct)

  def plotHeatmap(self, ax=None, is_plot=True, max_zero=5, figsize=(10, 12)):
    """
    Constructs a heatmap in which x-axis is state, y-axis is feature,
    value is significance level using a temperature color code.

    Parameters
    ----------
    ax: Matplotlib.axes
    is_plot: bool
    max_zero: float
    figsize: (float, float)

    Returns
    -------
    pd.DataFrame
        columns: class
        index: feature
        value: significance level
    """
    if ax is None:
      _, ax = plt.subplots(1, figsize=figsize)
    # Contruct a datadrame
    df = self.toDataframe()
    # Convert to number of zeros
    df = df.applymap(lambda v: util.convertSLToNumzero(v))
    df_plot = df.copy()
    df_plot.index = list(range(len(df_plot)))
    # Do the plot
    sns.heatmap(df_plot, cmap='seismic', ax=ax, vmin=-max_zero, vmax=max_zero)
    ax.set_ylabel = "feature vector"
    ax.set_xlabel = "class"
    #
    if is_plot:
      plt.show()
    return df

  def plotEvaluate(self, ser_X, max_sl=0.01, ax=None,
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
    if self.case_col is None:
      raise ValueError("Must do `build` first!")
    #
    feature_vector = FeatureVector(ser_X.to_dict())
    cases = [c for c in self.case_col.values()
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
      ser_plot = pd.Series([util.convertSLToNumzero(v) for v in ser_plot])
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
  def mkCaseCollectionQuery(cls, case_managers, **kwargs):
    """
    Constructs a CaseCollectionQuery.

    Parameters
    ----------
    case_managers: list-CaseManager
    kwargs: dict
    
    Returns
    -------
    CaseCollectionQuery
    """
    case_col_dct = [m.case_col for m in case_managers]
    return cls(case_col_dct, **kwargs)
