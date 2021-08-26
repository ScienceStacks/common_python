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

import copy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


################## FUNCTIONS ##############################
def checkKwargs(keywords, kwarg):
  """
  Verifies the presence of keywords.

  Parameters
  ----------
  keywords
  """
  if any([k is None for k in keywords]):
    raise RuntimeError("A keyword %s is unexpectedly None" % " ".join(keywords)


################## CLASSES ################################
class CaseQuery:
  """Selects Cases for a single class and displays results."""

  def __init__(self, case_col):
    """
    Parameters
    ----------
    case_col: CaseCollection
      
    """
    self.case_col = case_col
    if classes is None:
      classes = list(self.case_col.keys())
    self.classes = classes

  def toSeries(self):
    values = [c.siglvl for c in self.case_col.values()]
    return pd.Series(values, index=self.case_col.keys())

  def query(self, method, **kwargs):
    """
    Performs the requested query and returns a CaseQuery.

    Parameters
    ----------
    method: Method
        first argument: CaseCollection
        returns sorted CaseCollection
    kwargs: dict
        keyword arguments for method
    
    Returns
    -------
    CaseQuery
    """
    return CaseQuery(method(self.case_col, **kwargs))

  ######### STATIC METHOD QUERIES ###############
  @staticmethod
  def selectByDescription(case_col, ser_desc=None, terms=None):
    """
    Returns those cases that have at least one of the terms.

    Parameters
    ----------
    case_col: dict
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
        selected_cases.extend([c for c in self.case_col.values()
            if feature in c.feature_vector.fset.list])
      return list(set(selected_cases))
    #
    features = findFeaturesWithTerms(terms)
    return CaseCollection.make(findCasesWithFeature(features))

  @staticmethod
  def selectCaseByFeatureVector(case_col, feature_vector=None):
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
    checkKwargs(feature_vector)
    cases = [c for c in self.case_col.values()
        if c.feature_vector.isSubvector(feature_vector)]
    return CaseCollection.make(cases)


###########################################################
class CaseCollectionQuery:
  """Selects subsets of cases and displays results."""

  def __init__(self, case_cols, names=None):
    """
    Parameters
    ----------
    case_cols: list-CaseCollection
    names: list-str
    """
    self.case_cols = case_cols
    if names is None:
      names = [str(n) for n in range(len(case_cols))]
    self.names = names
    
  def toDataframe(self):
    """
    Converts case classes to a dataframe of significance levels.

    Returns
    -------
    pd.DataFrame
        index: feature_vector
        column: class
        value: significance level
    """
    sers = [ pd.Series([c.siglvl for c in d.values()], index=d.keys()])
        for d in self.case_cols]
    df = pd.concat(sers, axis=1)
    df.columns = list(self.self.case_cols.keys())
    return df

  def query(self, method, **kwargs):
    """
    Applies the query to each CaseCollection.

    Parameters
    ----------
    method: method that takes CaseCollection as first argument
    
    Returns
    -------
    CaseCollectionQuery
    """

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
  def mkCaseCollectionQuery(cls, case_managers, **kwargs)
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
    case_cols = [m.case_col for m in case_managers]
    return cls(case_cols, **kwargs)
