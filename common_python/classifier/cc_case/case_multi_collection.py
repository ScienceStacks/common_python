'''Interrogate and visualize a Collection of Cases for multiple clases'''

"""
A CaseMultiCollection is a collection of CaseCollections. Each
CaseCollection is built for a different class from the same
feature data.
"""

import common_python.constants as cn
import common_python.util.util as util
from common_python.classifier.feature_set import FeatureVector
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
    
  def toDataframe(self, max_sl=0.001):
    """
    Creates a dataframe of significance levels by class.

    Parameters
    ----------
    max_sl: float

    Returns
    -------
    pd.DataFrame
        index: feature_vector
        column: class
        value: significance level
    """
    def selDataframe(df, max_sl, direction):
      if direction < 0:
        sel = df[cn.SIGLVL] < 0
      else:
        sel = df[cn.SIGLVL] >= 0
      ser = direction*df[sel][cn.SIGLVL]
      if sum(sel) > 0:
        ser_result = ser[ser <= max_sl]
      else:
        ser_result  = pd.Series()
      return ser_result
    #
    def mkPrunedSer(collection, max_sl):
      df = collection.toDataframe()
      df_neg = selDataframe(df, max_sl, -1)
      df_pos = selDataframe(df, max_sl, 1)
      df_result = pd.concat([df_neg, df_pos], axis=0)
      return df_result
    # 
    sers = [mkPrunedSer(c, max_sl) for c in self.collection_dct.values()]
    df = pd.concat(sers, axis=1)
    df.columns = self.names
    #
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

  def plotBars(self, feature_vector=None, ax=None, is_plot=True,
      max_sl=0.001, expected_class=None,
      figsize=(5, 5), title="", fontsize=16,
      xticklabels=True, yticklabels=True,
      xlabel="class", ylabel="fraction positive"):
    """
    Constructs a bar plot of the fraction positive for each class.

    Parameters
    ----------
    feature_vector: FeatureVector
    expected_class: int
    ax: Matplotlib.axes
    is_plot: bool
    max_sl: float
    figsize: (float, float)
    """
    if ax is None:
      _, ax = plt.subplots(1, figsize=figsize)
    # Contruct a datadrame
    if feature_vector is not None:
      multi = self.select(CaseCollection.selectIsContained,
          feature_vector=feature_vector)
    else:
      multi = self
    fracs = []
    counts = []
    for name, collection in multi.collection_dct.items():
       frac, count = collection.countCases(max_sl=max_sl)
       fracs.append(frac)
       counts.append(count)
    # Do the plot
    bar_list = ax.bar(self.names, fracs)
    if expected_class is not None:
      bar_list[expected_class].set_color('r')
    for idx, frac in enumerate(fracs):
        ax.text(self.names[idx], frac + 0.01, str(counts[idx]),
            fontsize=fontsize)
    if not xticklabels:
      ax.set_xticklabels([])
    if not yticklabels:
      ax.set_yticklabels([])
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize+2)
    ax.set_ylim([0, 1.1])
    #
    if is_plot:
      plt.show()

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

  def serialize(self, multi_path, df_X_path=None, ser_y_path=None):
    """
    Serializes the CaseMultiCollection as a CSV file.

    Parameters
    ----------
    multi_path: str
        where to save CSV for collection
    df_X_path: str
        path to feature vector training data
    ser_y_path: str
        path to label training data
    """
    # Construct the dataframe to serialize
    dfs = [c.toDataframe() for c in self.collection_dct.values()]
    for name, df in zip(self.names, dfs):
      df[COL_KEY] = name
    df = pd.concat(dfs, axis=0)
    df.index.name = cn.INDEX
    #
    util.serializePandas(df, multi_path)

  @classmethod
  def deserialize(self, multi_path, df_X_path=None, ser_y_path=None):
    """
    Deserializes the collection from the path. The serializations is
    from a CSV file with a column added for the class.

    Parameters
    ----------
    multi_path: str
        where to save CSV for collection
    df_X_path: str
        path to feature vector training data
    ser_y_path: str
        path to label training data

    Returns
    -------
    CaseMultiCollection
    """
    # Obtain the dataframe representation
    df = pd.read_csv(multi_path)
    df = df.set_index(cn.INDEX)
    names = list(set(df[COL_KEY]))
    df_dct = {n: df[df[COL_KEY] == n] for n in names}
    for df in df_dct.values():
      del df[COL_KEY]
    collection_dct = {n:
        CaseCollection.deserialize(df=df, df_X_path=df_X_path,
        ser_y_path=ser_y_path)
        for n, df in df_dct.items()}
    return CaseMultiCollection(collection_dct)

  def plotBarsForFeatures(self, df_X, num_row, num_col,
     ser_y=None, suptitle="", **kwargs):
    """
    Does bar plots for a matrix of feature data.

    Parameters
    ----------
    df_X: pd.DataFrame
        rows: instances
        columns: features
    num_row: int
         number of rows of plots
    num_col: int
         number of columns of plots
    ser_y: pd.Series
        expected class
    suptitle: str
    kwargs: dict
         key words provided to plotBars
    
    Returns
    -------
    """
    if "figsize" in kwargs:
      figsize = kwargs["figsize"]
    else:
      figsize = (10, 12)
    if "fontsize" in kwargs:
      fontsize = kwargs["fontsize"]
    else:
      fontsize = 8
      kwargs["fontsize"] = fontsize
    #
    if (num_row == 1) and (num_col == 1):
      fig, axes = plt.subplots(1, figsize=figsize)
    else:
      fig, axes = plt.subplots(num_row, num_col, figsize=figsize)
    indices = list(df_X.index)
    try:
      indices = sorted(indices, key=lambda s: float(s[1:3]))
    except Exception:
      pass
    for irow in range(num_row):
      for icol in range(num_col):
        idx = irow*num_col + icol
        instance = indices[idx]
        feature_vector = FeatureVector(df_X.loc[instance, :])
        if isinstance(axes, np.ndarray):
          ax = axes[irow, icol]
        else:
          ax = axes
        if ser_y is not None:
          expected_class = ser_y.loc[instance]
        else:
          expected_class = None
        self.plotBars(feature_vector=feature_vector, is_plot=False,
            title=instance, ax=ax, expected_class=expected_class,
            xlabel="", ylabel="", xticklabels=False, yticklabels=False, **kwargs)
        plt.suptitle(suptitle)
    plt.show()
    if "is_plot" in kwargs:
      if not kwargs["is_plot"]:
        pass
      else:
        plt.show()
