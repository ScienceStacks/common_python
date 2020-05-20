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
import numpy as np
import os
import pandas as pd
from sklearn import svm
import threading


DIR = os.path.dirname(os.path.abspath(__file__))
PERSISTER_PATH = os.path.join(DIR,
    "feature_eqivalence_calculator.pcl")
SELECTED_FEATURE = "selected_feature"
ALTERNATIVE_FEATURE = "alternative_feature"
NUM_CROSS_ITER = 50


class FeatureEquivalenceCalculator(object):

  def __init__(self,
      df_X,
      ser_y,
      base_clf=svm.LinearSVC(),
      num_holdouts=1,
      num_cross_iter=NUM_CROSS_ITER,
      persister=Persister(PERSISTER_PATH),
      min_score=0.9,
      is_restart=True,
      ):
    """
    :param pd.DataFrame df_X: feature vector
    :param pd.Series ser_y: binary classes
    :param list-object features: set of possible features
    :param Classifier base_clf:  base classifier
        Exposes: fit, score, predict
    :param float min_score: minimum score for a FitResult
    """
    if is_restart:
      ########### PRIVATE ##########
      self._persister = persister
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

  def _checkpoint(self):
    self._persister.set(self)

  def run(self, fit_results):
    """
    Calculates feature equivalences based on the feature sets
    provided.   
    :param list-mcfo.FitResult fit_results:
    :param pd.DataFrame df_X:
        columns: featuress
        index: features
        values: m(F, f1, f2)
    Places the following dataframe in self.ria_dct
      Columns: SELECTED_FEATURE, ALTERNATIVE_FEATURE
      Values: m(F, f1, f2)
    """
    # Find all features in the fit results
    all_features = []
    [all_features.extend(fr.sels) for fr in fit_results]
    for fit_result in fit_results:
      # Get the dictionary for calculating RIA
      if not fit_result.idx in self.ria_dct.keys():
        ria_dct =  {
            SELECTED_FEATURE: [],
            ALTERNATIVE_FEATURE: [],
            cn.SCORE: [],
            }
      else:
        keys = [SELECTED_FEATURE, ALTERNATIVE_FEATURE,
            cn.SCORE]
        ria_dct = {}
        for key in keys:
          ria_dct[key] = self.ria_dct[
              fit_result.idx][key].tolist()
      length = len(all_features)
      # Process each selected feature
      for selected_feature in fit_result.sels:
        ria_dct[SELECTED_FEATURE].extend(np.repeat(
            selected_feature, length).tolist())
        ria_dct[ALTERNATIVE_FEATURE].extend(
            all_features)
        ria_dct[cn.SCORE].extend(
            self._calculateRIA(selected_feature,
            fit_result.sels, all_features))
        self._checkpoint()  # checkpoint acquires lock
      # Save the result
      df = pd.DataFrame(ria_dct)
      self.ria_dct[fit_result.idx] = df

  def _calculateRIA(self, selected_feature, 
      selected_features, alternative_features):
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
        self._df_X[selected_features], self._ser_y,
        partitions=self._partitions)
    candidates = list(selected_features)
    candidates.remove(selected_feature)
    score_without_sel =  \
        util_classifier.binaryCrossValidate(
        self._base_clf,
        self._df_X[candidates], self._ser_y,
        partitions=self._partitions)
    denom = score_with_sel - score_without_sel
    score_rias = []
    for candidate in alternative_features:
      # Calculate the relative incremental accuracy
      candidates.append(candidate)
      score_with_alt =  \
          util_classifier.binaryCrossValidate(
          self._base_clf,
          self._df_X[candidates], self._ser_y,
          partitions=self._partitions)
      candidates.remove(candidate)
      numr = score_with_alt - score_without_sel
      if np.isclose(denom, 0):
        score_ria = np.nan
      else:
        score_ria = numr / denom
      score_rias.append(score_ria)
    return score_rias
