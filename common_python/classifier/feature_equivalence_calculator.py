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
relative incremental accuracy (RIA):
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
NUM_CROSS_ITER = 100


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
      num_cross_iter=NUM_CROSS_ITER,
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
      self._partitions =  \
          [util_classifier.partitionByState(
          ser_y, holdouts=num_holdouts)
          for _ in range(num_cross_iter)]
      ########### PUBLIC ##########
      self.ria_dct = {}  # key: idx, value: df

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
    Places the following dataframe in self.ria_dct
      Columns: SELECTED_FEATURE, ALTERNATIVE_FEATURE
      Values: m(F, f1, f2)
    """
    logging.info("Start processing SFS %d" % fit_result.idx)
    # Get the dictionary for calculating RIA
    LOCK.acquire()
    if not fit_result.idx in self.ria_dct.keys():
      ria_dct[fit_result.idx] =  {
          SELECTED_FEATURE: [],
          ALTERNATIVE_FEATURE: [],
          cn.SCORE: [],
          }
      self.ria_dct[fit_result.idx] = ria_dct
    else:
      ria_dct = self.ria_dct[fit_result.idx]
    LOCK.release()
    #
    # Features to compare against
    alternative_features = set(
        self._features).difference(
        ria_dct[fit_result.idx][ALTERNATIVE_FEATURE])
    length = len(alternative_features)
    # Process each selected feature
    for selected_feature in fit_result.sels:
      ria_dct[SELECTED_FEATURE].extend(list(
          np.repeat(selected_feature, length)))
      ria_dct[ALTERNATIVE_FEATURE].extend(
          alternative_features)
      ria_dct[cn.SCORE].extend(
          self._calculateRIA(selected_feature,
          fit_result.sels, alternative_features))
      self.checkpoint()  # checkpoint acquires lock
    # Save the result
    df = pd.DataFrame(ria_dct)
    LOCK.acquire()
    self.ria_dct[fit_result.idx] = df
    LOCK.release()
    logging.info("Completed processing SFS %d" % fit_result.idx)

  def _calculateRIA(self, selected_feature, 
      seleted_features, alternative_features):
    """
    Calculates RIA (relative incremental accuracy) for
    features in a classification group (set of features
    used for a classifier)
    :param object feature:
    :param list-object selected_features:
        features in the RIA
    :param list-object alternative_features:
        features to compare with
    :return list-float: scores for the selected feature
    """
    score_with_sel =  \
        util_classifier.binaryCrossValidate(
        self._base_clf,
        self._df_X[selected_feature], self._ser_y,
        partitions=self._partitions)
    candidate_features = list(selected_features)
    candidate_features.remove(selected_feature)
    score_without_sel =  \
        util_classifier.binaryCrossValidate(
        self._base_clf,
        self._df_X[without_feature], self._ser_y,
        partitions=self._partitions)
    score_incr_sel =  \
        score_with_sel - score_without_sel
    score_rias = []
    for alternative_feature in alternative_features:
      # Calculate the relative incremental accuracy
      candidate_features.append(alternative_feature)
      score_with_alt = util_classifier.scoreFeatures(
          self._base_clf,
          df_X[candidate_features], ser_y,
          partitions=self._partitions)
      candiate_feature.remove(alternative_feature)
      score_rias.append((score_with_alt
          - score_without_sel)  / score_incr_sel)
    return score_rias
