'''Interrogate and visualize a CaseCollection for multiple clases'''

"""
A CaseMultiCollection is a collection of CaseCollections. Each
CaseCollection is built for a different class from the same
feature data.
"""

import copy
import numpy as np
import pandas as pd


###########################################################
class CaseMultiCollection:
  """Interrogates and displays cases for multiple classes."""

  def __init__(self, dct):
    """
    Parameters
    ----------
    dct: dict
        key: collection identifier (e.g., class name)
        value: CaseCollection
    """
    self.names = list(dct.keys())
    self.collection_dct = dict(dct)
    
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
    sers = [c.toDataframe()[cn.SIGLVL] for c in self.collection_dct.values()]
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
    CaseMultiCollection
    """
    dct = {}
    for name, case_col in self.collection_dct.items():
      dct[name] = CaseCollection.select(method, case_col, **kwargs)
    return CaseMultiCollection(dct)

  def binarySelect(self, other_multi, method, **kwargs):
    """
    Does selection for two CaseMultiCollections (e.g., union, intersection)

    Parameters
    ----------
    method: method that takes CaseCollection as the first 2 arguments
    other_multi: CaseMultiCollection
    
    Returns
    -------
    CaseMultiCollection
    """
    dct = {}
    for name, case_col in self.collection_dct.items():
      dct[name] = CaseCollection.select(method, case_col,
          multi_collection.collection_dct[name], **kwargs)
    return CaseMultiCollection(dct)

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
  def mkCaseMultiCollection(cls, case_managers, **kwargs):
    """
    Constructs a CaseMultiCollection.

    Parameters
    ----------
    case_managers: list-CaseManager
    kwargs: dict
    
    Returns
    -------
    CaseMultiCollection
    """
    case_col_dct = [m.case_col for m in case_managers]
    return cls(case_col_dct, **kwargs)
