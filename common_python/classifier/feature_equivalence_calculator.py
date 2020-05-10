'''Estimate the equivalence of two features.'''

"""
Two classification features are equivalent if result in
the same classification accuracy when they are exchanged
in the set of features for the classifier.

Let a(F) be the accuracy of a classifier trained on the
features F. Let F-f be F without the feature f, and
F+f be F with the feature f.

FeatureEquivalenceCalculator calculates the metric
m(F, f1, f2), the incremental accuracy of using f1
compared if the incremental accuracy of using f2.
m(F, f1, f2) = (A(F-f1+f2) - A(F-f1))/(A(F) - A(F-f1)).

Since ths is computationally intensive, the calculator is
restartable and multi-threaded.
"""

import common_python.constants as cn
from common_python.classifier  \
    import multi_classifier_feature_optimizer as mcfo
from common_python.classifier import util_classifier
from common_python.classifier.feature_collection  \
    import FeatureCollection
from common_python.util.persister import Persister

import collections
import concurrent.futures
import copy
import logging
import numpy as np
import os
import pandas as pd
from sklearn import svm
import threading


# Default checkpoint callback
DIR = os.path.dirname(os.path.abspath(__file__))
PERSISTER_PATH = os.path.join(DIR,
    "feature_eqivalence_calculator.pcl")
LOCK = threading.Lock()
MAX_WORKER_THREADS = 6
SELECTED_FEATURE = "selected_feature"
ALTERNATIVE_FEATURE = "alternative_feature"


FitResult = collections.namedtuple("FitResult",
    "idx sels sels_score all_score excludes n_eval")
#    idx: index of the interaction of excludes
#    sels: list of features selected
#    sels_score: score for classifier with selects
#    all_score: score for classifier with all non-excludes
#    excludes: list of features excluded
#    n_eval: number of features evaluated


class FeatureEquivalenceCalculator(object):

  def __init__(self,
      df_X,
      ser_y,
      base_clf=svm.LinearSVC(),
      is_restart=True,
      ):
    """
    :param FeatureCollection feature_collection:
    :param list-object features: set of possible features
    :param Classifier base_clf:  base classifier
        Exposes: fit, score, predict
    """
    if persister is None:
      self._persister = Persister(PERSISTER_PATH)
    else:
      self._persister = persister
    if is_restart:
      ########### PRIVATE ##########
      self._base_clf = copy.deepcopy(base_clf)
      self._df_X = df_X
      self._ser_y = ser_y
      ########### PUBLIC ##########
      self.equiv_dct = {}  # key: idx, value: df

  def checkpoint(self):
    LOCK.acquire()
    self._persister.set(self)
    LOCK.release()

  def run(self, fit_results):
    """
    Calculates feature equivalences based on the feature sets
    provided.   
    :param list-mcfo.FitResult fit_results:
    :param pd.DataFrame df_X:
        columns: featuress
        index: features
        values: m(F, f1, f2)
    """
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, 
        level=logging.INFO, datefmt="%H:%M:%S")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_WORKER_THREADS) as executor:
      executor.map(self._run_one, fit_results)
    # Merge the results into a single DataFrame

  def _run_one(self, fit_result):
    """
    Calculates metrics for one set of features constructed
    by a fitter.
    Places the following dataframe in self.equiv_dct
      Columns: SELECTED_FEATURE, ALTERNATIVE_FEATURE
      Values: m(F, f1, f2)
    """
    logging.info("Start processing class %d" % fit_result.idx)
    LOCK.acquire()
    if not fit_result.idx in self.equiv_dct.keys():
      equiv_dct[fit_resultt.idx] =  {
          SELECTED_FEATURE: [],
          ALTERNATIVE_FEATURE: [],
          cn.SCORE: [],
          }
      self.equiv_dct[fit_result.idx] = equiv_dct
    else:
      equiv_dct = self.equiv_dct[fit_result.idx]
    LOCK.release()
    alternative_features = set(
        self._df_X.columns).difference(
        equiv_dct[fit_result.idx][ALTERNATIVE_FEATURE])
    collection = FeatureCollection(self._df_X, self._ser_y)
    [collection.add(feature=f) for f in fit_result.sels]
    for selected_feature in fit_result.sels:
      current_collection = copy.deepcopy(collection)
      current_collection.remove(selected_feature)
      for alternative_feature in alternative_features:
        current_collection.add(alternative_feature)
        # Evaluate, compute metric, record result
        current_collection.remove(alternative_feature)
      self.checkpoint()  # checkpoint acquires lock
    logging.info("Completed processing class %d" % cl)
