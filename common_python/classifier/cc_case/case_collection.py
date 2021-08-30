'''Interrogate and visualize a collection of cases'''

"""
A CaseCollection is a collection of Cases that are accessed by
their FeatureValue string. The class exposes ways to interrogate cases
and construct new cases.
"""

import common_python.constants as cn
import common_python.util.util as util
from common_python.classifier.feature_set import FeatureVector
from common_python.classifier.cc_case.case_core  \
    import FeatureVectorStatistic, Case

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAX_SL = 0.05
MIN_SL = 1e-5  # Minimum value of significance level used
IS_CHECK = True  # Does additional tests of consistency


################## FUNCTIONS ##############################
def checkKwargs(keyword_values):
  """
  Verifies the presence of keywords.

  Parameters
  ----------
  keyword_values: list-object
  """
  if any([v is None for v in keyword_values]):
    raise RuntimeError("A keyword %s is unexpectedly None" %
        " ".join(keyword_values))


##################################################################
class CaseCollection(dict):

  version = 3

  """
  key: FeatureVector in sorted order
  value: Case
  """
  def __init__(self, dct, df_X=None, ser_y=None):
    """
    Parameters
    ----------
    dct: dict
    df_X: pd.DataFrame
        feature vector training data
    ser_y: pd.Series
        label training data
    """
    super().__init__(dct)
    self.df_X = df_X
    self.ser_y = ser_y

  def sort(self):
    """
    Sorts the dictionary by key.
    """
    keys = list(self.keys())
    # Check if a sort is needed
    is_sorted = True
    for key1, key2 in zip(keys[0:-1], keys[1:]):
      if key1 > key2:
        is_sorted = False
        break
    if is_sorted:
      return
    #
    keys.sort()
    dct = {k: self[k] for k in keys}
    _  = [self.__delitem__(k) for k in keys]
    self.update(dct)

  def __eq__(self, other_col):
    def check(pd_obj1, pd_obj2):
      if isinstance(pd_obj1, pd.DataFrame):
        return pd_obj1.equals(pd_obj2)
      elif isinstance(pd_obj2, pd.DataFrame):
        return pd_obj2.equals(pd_obj2)
      elif isinstance(pd_obj1, pd.Series):
        return pd_obj1.equals(pd_obj2)
      elif isinstance(pd_obj2, pd.Series):
        return pd_obj2.equals(pd_obj2)
      return pd_obj1 == pd_obj2
    #
    result = True
    result = result and check(self.df_X, other_col.df_X)
    result = result and check(self.ser_y, other_col.ser_y)
    diff = set(self.keys()).symmetric_difference(other_col.keys())
    result = result and (len(diff) == 0)
    result = result and all([v == other_col[k] for k,v in self.items()])
    return result

  def toDataframe(self):
    """
    Creates a dataframe from the data in the cases.

    Returns
    -------
    pd.DataFrame
        index: str(feature_vector)
        columns: cn.NUM_POS, cn.NUM_POS, cn.SIGLVL
    """
    siglvls = [c.fv_statistic.siglvl for c in self.values()]
    num_samples = [c.fv_statistic.num_sample for c in self.values()]
    num_poss = [c.fv_statistic.num_pos for c in self.values()]
    prior_probs = [c.fv_statistic.prior_prob for c in self.values()]
    instance_strs = [c.instance_str for c in self.values()]
    df = pd.DataFrame({
        cn.SIGLVL: siglvls,
        cn.PRIOR_PROB: prior_probs,
        cn.NUM_SAMPLE: num_samples,
        cn.NUM_POS: num_poss,
        cn.INSTANCE_STR:  instance_strs,
        }, index=list(self.keys()))
    return df.sort_index()

  def countCases(self, is_drop_duplicate=True):
    """
    Counts the positive and negative cases that are not redundant in the
    training data (in that they select difference instances).
    
    Returns
    -------
    float, int
        fraction positive cases, total cases
    is_drop_duplicate: bool
        drop cases that are for the same samples
    """
    # Construct dataframe that eliminates feature vectors that select
    # the same instances
    df = self.toDataframe()
    df[cn.FRAC] = df[cn.NUM_POS] / df[cn.NUM_SAMPLE]
    if is_drop_duplicate:
      df = df.set_index(cn.INSTANCE_STR)
      df = df.drop_duplicates()
    # Calculate statistics
    num_tot = len(df)
    sel_neg = df[cn.SIGLVL] < 0
    if max(df[sel_neg][cn.FRAC] > 0.9):
      raise RuntimeError("Positive example is classified as negative.")
    sel_pos = df[cn.SIGLVL] > 0
    if min(df[sel_pos][cn.FRAC] < 0.1):
      raise RuntimeError("Negative example is classified as positive.")
    return sum(sel_pos)/num_tot, num_tot
    

  @staticmethod
  def _checkCommon(case_col1, case_col2):
    if IS_CHECK:
      common_stg = list(set(case_col1.keys()).intersection(case_col2.keys()))
      trues = [case_col1[k] == case_col2[k] for k in common_stg]
      if not all(trues):
        raise RuntimeError("Common Cases are not equal.")

  #################### Case Selection ######################
  # All methods must be static and first argument is CaseCollection
  @staticmethod
  def selectUnion(case_col, other_col=None):
    """
    Union of two CaseCollection.

    Parameters
    ----------
    case_col: CaseCollection
    other_col: CaseCollection

    Returns
    -------
    CaseCollection
    """
    checkKwargs([other_col])
    CaseCollection._checkCommon(case_col, other_col)
    new_case_col = copy.deepcopy(case_col)
    new_case_col.update(other_col)
    new_case_col.sort()
    return new_case_col

  @staticmethod
  def selectIntersection(case_col, other_col=None):
    """
    Intersection of two CaseCollection.

    Parameters
    ----------
    case_col: CaseCollection
    other_col: CaseCollection

    Returns
    -------
    CaseCollection
    """
    checkKwargs([other_col])
    CaseCollection._checkCommon(case_col, other_col)
    common_keys = set(case_col.keys()).intersection(other_col.keys())
    cases = [case_col[k] for k in common_keys]
    return CaseCollection.make(cases, df_X=case_col.df_X, ser_y=case_col.ser_y)

  @staticmethod
  def selectDifference(case_col, other_col):
    """
    Difference of two CaseCollection.

    Parameters
    ----------
    case_col: CaseCollection
    other_col: CaseCollection

    Returns
    -------
    CaseCollection
    """
    checkKwargs([other_col])
    CaseCollection._checkCommon(case_col, other_col)
    difference_keys = set(case_col.keys()).difference(other_col.keys())
    cases = [case_col[k] for k in difference_keys]
    return CaseCollection.make(cases, df_X=case_col.df_X, ser_y=case_col.ser_y)

  @staticmethod
  def selectSymmetricDifference(case_col, other_col):
    """
    What's not common to both.

    Parameters
    ----------
    case_col: CaseCollection
    other_col: CaseCollection

    Returns
    -------
    CaseCollection
    """
    checkKwargs([other_col])
    CaseCollection._checkCommon(case_col, other_col)
    new_case_col = case_col.difference(other_col)
    new_case_col.update(other_col.difference(case_col))
    return new_case_col

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
      return list(case_col.values())
    #
    def findFeaturesWithTerms(terms):
      """
      Finds features with descriptions that contain at least one term.

      Parameters
      ----------
      terms: list-str

      Returns
      -------
      list-str
          string representation of feature (str)
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
      features: list-str
          list of feature (str)

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
    cases = findCasesWithFeature(features)
    return CaseCollection.make(cases, df_X=case_col.df_X, ser_y=case_col.ser_y)

  @staticmethod
  def selectByFeatureVector(case_col, feature_vector=None):
    """
    Finds cases that contain the Feature Vector.

    Parameters
    ----------
    case_col: CaseCollection
    feature_vector: FeatureVector

    Returns
    -----
    CaseCollection
    """
    checkKwargs([feature_vector])
    cases = [c for c in case_col.values()
        if c.feature_vector.contains(feature_vector)]
    return CaseCollection.make(cases, df_X=case_col.df_X, ser_y=case_col.ser_y)

  @staticmethod
  def selectIsContained(case_col, feature_vector=None):
    """
    Finds cases that are contained in the Feature Vector.

    Parameters
    ----------
    case_col: CaseCollection
    feature_vector: FeatureVector

    Returns
    -----
    CaseCollection
    """
    checkKwargs([feature_vector])
    cases = [c for c in case_col.values()
        if feature_vector.contains(c.feature_vector)]
    return CaseCollection.make(cases, df_X=case_col.df_X, ser_y=case_col.ser_y)

  ################## Instance versions of the queries
  def union(self, other_col):
    return CaseCollection.selectUnion(self, other_col=other_col)

  def intersection(self, other_col):
    return CaseCollection.selectIntersection(self, other_col=other_col)

  def difference(self, other_col):
    return CaseCollection.selectDifference(self, other_col=other_col)

  def symmetricDifference(self, other_col):
    return CaseCollection.selectSymmetricDifference(self, other_col=other_col)

  def findByDescription(self, ser_desc, terms):
    return CaseCollection.selectByDescription(self, ser_desc=ser_desc,
        terms=terms)

  def findByFeatureVector(self, feature_vector):
    return CaseCollection.selectByFeatureVector(self,
        feature_vector=feature_vector)

  def findIsContained(self, feature_vector):
    return CaseCollection.selectIsContained(self,
        feature_vector=feature_vector)

  def serialize(self, collection_path, df_X_path=None, ser_y_path=None):
    """
    Serializes the CaseCollection as a CSV file.

    Parameters
    ----------
    collection_path: str
        where to save CSV for collection
    df_X_path: str
        path to feature vector training data
    ser_y_path: str
        path to label training data
    """
    util.serializePandas(self.toDataframe(), collection_path)
    if df_X_path is not None:
      if not os.path.isfile(df_X_path):
        util.serializePandas(self.df_X)
    if ser_y_path is not None:
      if not os.path.isfile(ser_y_path):
        util.serializePandas(self.ser_y)

  @classmethod
  def deserialize(cls, collection_path=None, df=None,
      df_X_path=None, ser_y_path=None):
    """
    Deserializes the CaseCollection from a CSV file.

    Parameters
    ----------
    collection_path: str
      Path to deserialize from, a CSV file with a column named "index"
    df: DataFrame
      A dataframe representation of the cases to construct
        index: str representation of feature vector
    df_X_path: str
        path to feature vector training data
    ser_y_path: str
        path to label training data

    Returns
    -------
    CaseCollection
    """
    # Retrieve the files
    if (collection_path is not None) and (df is None):
      df = util.deserializePandas(collection_path)
    elif (collection_path is None) and (df is not None):
      pass
    else:
     raise ValueError("Exactly one of collection_path and df must be non-None.")
    if df_X_path is not None:
      df_X = util.deserializePandas(df_X_path)
    else:
      df_X = None
    if ser_y_path is not None:
      ser_y = util.deserializePandas(ser_y_path)
    else:
      ser_y = None
    #
    case_dct = {}
    for fv_str, row in df.iterrows():
      feature_vector = FeatureVector.make(fv_str)
      statistic = FeatureVectorStatistic(
          row[cn.NUM_SAMPLE],
          row[cn.NUM_POS],
          row[cn.PRIOR_PROB],
          row[cn.SIGLVL])
      case_dct[fv_str] = Case(feature_vector, statistic, df_X=df_X)
    #
    return CaseCollection(case_dct, df_X=df_X, ser_y=ser_y)

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
    CaseCollection
        cases plotted
    """
    # Select applicable cases
    feature_vector = FeatureVector(ser_X.to_dict())
    cases = [c for c in self.values()
         if feature_vector.contains(c.feature_vector)
         and np.abs(c.fv_statistic.siglvl) < max_sl]
    feature_vectors = [c.feature_vector for c in cases]
    siglvls = [c.fv_statistic.siglvl for c in cases]
    count = len(cases)
    # Construct plot Series
    # Do the plot
    if is_plot and (count == 0):
        print("***No Case found for %s" % title)
    else:
      ser_plot = pd.Series(siglvls, dtype="float64")
      ser_plot.index = ["" for _ in range(count)]
      labels  = [str(c) for c in feature_vectors]
      ser_plot = pd.Series([util.convertSLToNumzero(v) for v in ser_plot],
          dtype='float64')
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
    return CaseCollection.make(cases, df_X=self.df_X, ser_y=self.ser_y)

  ################### Other CLASS METHODS ###################
  @classmethod
  def make(cls, cases, df_X=None, ser_y=None):
    """
    Returns sorted CaseCollection.

    Parameters
    ----------
    cases: list-Case
    df_X: pd.DataFrame
        feature vector training data
    ser_y: pd.Series
        label training data

    Returns
    -------
    CaseCollection (sorted)
    """
    dct = {str(c.feature_vector): c for c in cases}
    case_col = CaseCollection(dct, df_X=df_X, ser_y=ser_y)
    case_col.sort()
    return case_col
