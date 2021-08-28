'''Interrogate and visualize a Collection of Cases for multiple clases'''

"""
A CaseMultiCollection is a collection of CaseCollections. Each
CaseCollection is built for a different class from the same
feature data.
"""

import common_python.constants as cn
import common_python.util.util as util
from common_python.classifier.cc_case.case_collection import CaseCollection

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

COL_KEY = "__col_key__"  # Column for the key in serialized CSV


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

  def __len__(self):
    return sum([len(c) for c in self.collection_dct.values()])

  def __eq__(self, other):
    diff_names = set(self.names).symmetric_difference(other.names)
    if len(diff_names) > 0:
      return False
    result = True
    for name in self.names:
      result = result and (
          self.collection_dct[name] == other.collection_dct[name])
    return result
    
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
      dct[name] = method(case_col, **kwargs)
    return CaseMultiCollection(dct)

  def binarySelect(self, method, other_multi, **kwargs):
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
      dct[name] = method(case_col,
          other_multi.collection_dct[name], **kwargs)
    return CaseMultiCollection(dct)

  def plotBars(self, ser_X, title="", **kwargs):
    """
    Creates a classification profile for the feature vector.
    
    Parameters
    ----------
    ser_X: Series (feature vector)
    """
    new_kwargs = dict(kwargs)
    new_kwargs["is_plot"] = False
    num_row = 2 
    num_col = 3 
    fig, axes = plt.subplots(num_row, num_col,
        figsize=(16, 10))
    for idx, name in enumerate(self.names):
        row = int(idx/num_col)
        col = idx % num_col
        if row == 0:
            label_xoffset = -0.1
        else:
            label_xoffset = 0.1 
        self.collection_dct[name].plotEvaluate(ser_X, 
            ax=axes[row, col],
            title = name,
            label_xoffset=label_xoffset, **new_kwargs)
    fig.suptitle(title, fontsize=16)
    if "is_plot" in kwargs:
      if kwargs["is_plot"]:
        plt.show()

  def plotHeatmap(self, feature_vector=None, ax=None, is_plot=True,
      max_zero=5, figsize=(10, 12), title=""):
    """
    Constructs a heatmap in which x-axis is state, y-axis is feature,
    value is significance level using a temperature color code.
    If feature_vector is not None, then only considers cases
    that are contained within this vector.

    Parameters
    ----------
    feature_vector: FeatureVector
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
    if feature_vector is not None:
      multi = self.select(CaseCollection.selectIsContained,
          feature_vector=feature_vector)
    else:
      multi = self
    df = multi.toDataframe()
    df = df.applymap(lambda v: util.convertSLToNumzero(v))
    df_plot = df.copy()
    df_plot.index = list(range(len(df_plot)))
    # Do the plot
    sns.heatmap(df_plot, cmap='seismic', ax=ax, vmin=-max_zero, vmax=max_zero)
    ax.set_ylabel("feature vector")
    ax.set_xlabel("class")
    ax.set_title(title)
    #
    if is_plot:
      plt.show()
    return df

  @classmethod
  def make(cls, case_builder_dct, **kwargs):
    """
    Constructs a CaseMultiCollection.

    Parameters
    ----------
    case_builder_dct: dict
        key: builder ID
        value: CaseBuilder
    kwargs: dict
    
    Returns
    -------
    CaseMultiCollection
    """
    case_col_dct = {k: m.case_col for k, m in case_builder_dct.items()}
    return cls(case_col_dct, **kwargs)

  def serialize(self, path):
    """
    Serializes the collection to the path. The serializations is a
    CSV file with a column added for the class.

    Parameters
    ----------
    
    Returns
    -------
    """
    # Construct the dataframe to serialize
    dfs = [c.toDataframe() for c in self.collection_dct.values()]
    for name, df in zip(self.names, dfs):
      df[COL_KEY] = name
    df = pd.concat(dfs, axis=0)
    df.index.name = cn.INDEX
    df.to_csv(path, index=True)

  @classmethod
  def deserialize(self, path):
    """
    Deserializes the collection from the path. The serializations is
    from a CSV file with a column added for the class.

    Parameters
    ----------
    
    Returns
    -------
    """
    # Obtain the dataframe representation
    df = pd.read_csv(path)
    df = df.set_index(cn.INDEX)
    names = list(set(df[COL_KEY]))
    df_dct = {n: df[df[COL_KEY] == n] for n in names}
    for df in df_dct.values():
      del df[COL_KEY]
    collection_dct = {n: CaseCollection.deserialize(df=df)
        for n, df in df_dct.items()}
    return CaseMultiCollection(collection_dct)
