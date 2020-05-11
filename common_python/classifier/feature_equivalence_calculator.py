'''Estimate the equivalence of two features.'''

"""
The intent of this analysis is to evaluate the
features in a Sufficient Feature Set (SFS), features that
are sufficient to construct a high accuracy classifier.
Two classification features are equivalent if they have
the same effect on classification accuracy for a SFS.

Let a(F) be the accuracy of a classifier trained on the
features F. Let F-f be F without the feature f, and
F+f be F with the feature f.

FeatureEquivalenceCalculator calculates the metric
relative incremental accuracy:
m(F, f1, f2), the incremental accuracy of using f2
compared to the incremental accuracy of using f1.
m(F, f1, f2) = (a(F-f1+f2) - a(F-f1))/(a(F) - a(F-f1)).
f1 is equivalent to f2 for F if m(F, f1, f2) ~ 1.

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
      num_holdouts=1,
      is_restart=True,
      ):
    """
    :param pd.DataFrame df_X: feature vector
    :param pd.Series ser_y: binary classes
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
      self._num_holdouts = num_holdouts
      self._base_clf = copy.deepcopy(base_clf)
      self._df_X = df_X
      self._ser_y = ser_y
      self._features = self._df_X.columns.tolist()
      self._num_holdouts = None
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
    logging.info("Start processing SFS %d" % fit_result.idx)
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
        self._features).difference(
        equiv_dct[fit_result.idx][ALTERNATIVE_FEATURE])
    collection = FeatureCollection(self._df_X, self._ser_y)
    [collection.add(feature=f) for f in fit_result.sels]
    current_collection = copy.deepcopy(collection)
    score_base = fit_result.sels_score
    for selected_feature in fit_result.sels:
      current_collection.remove(selected_feature)
      score_without = util_classifier.scoreFeatures(
          self._base_clf,
          df_X[collection.chosensi], ser_y,
          train_idxs=None, test_idxs=None):
      for alternative_feature in alternative_features:
        # Calculate the relative incremental accuracy
        current_collection.add(alternative_feature)
        score_with = util_classifier.scoreFeatures(
            self._base_clf,
            df_X[collection.chosensi], ser_y,
            train_idxs=None, test_idxs=None):
        current_collection.remove(alternative_feature)
        current_collection.add(selected_feature)
        relative_score = (score_with - score_without)  \
            / (score_base - score_without)
        equiv_dct[SELECTED_FEATURE].append(selected_feature)
        equiv_dct[ALTERNATIVE_FEATURE].append(
            alternative_feature)
        equiv_dct[cn.SCORE].append(relative_score)
        current_collection.remove(alternative_feature)
      self.checkpoint()  # checkpoint acquires lock
    logging.info("Completed processing SFS %d" % fit_result.idx)

  # FIXME: select training and tests indices
  def _calculateForSelectedFeature(self,
    base_score, collection, seleted_feature,
    alternative_features, equiv_dct):
    """
    """
      self._partitions =  \
          [util_classifier.partitionByState(
          ser_y, holdouts=self._num_holdouts)
          for _ in range(len(alternative_feature))]
      score_without = util_classifier.scoreFeatures(
          self._base_clf,
          self._df_X[collection.chosens], self._ser_y,
          train_idxs=None, test_idxs=None):
      for alternative_feature in alternative_features:
        # Calculate the relative incremental accuracy
        collection.add(alternative_feature)
        score_with = util_classifier.scoreFeatures(
            self._base_clf,
            df_X[collection.chosensi], ser_y,
            train_idxs=None, test_idxs=None):
        collection.remove(alternative_feature)
        collection.add(selected_feature)
        relative_score = (score_with - score_without)  \
            / (score_base - score_without)
        equiv_dct[SELECTED_FEATURE].append(selected_feature)
        equiv_dct[ALTERNATIVE_FEATURE].append(
            alternative_feature)
        equiv_dct[cn.SCORE].append(relative_score)
        collection.remove(alternative_feature)
