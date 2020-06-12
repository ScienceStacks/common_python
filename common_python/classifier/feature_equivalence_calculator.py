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
f1 is equivalent to f2 for F if m(F, f1, f2) = 1.
A deviaton from 1 indicates a poor substitute. An
adjusted value in the range [0, 1] is calculated
exp(-(1 - abs(m(F, f1, f2))).

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
NUM_CROSS_ITER = 50
NO_FEATURE_SCORE = 0.5


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
      self.df_ria = pd.DataFrame()

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
    Places the following dataframe in self.df_ria
      Columns: cn.CLS_FEATURE, cn.CMP_FEATURE
      Values: m(F, f1, f2)
    """
    # Find all features in the fit results
    all_features = []
    [all_features.extend(fr.sels) for fr in fit_results]
    if len(self.df_ria) == 0:
      processed_features = []
    else:
      processed_features = self.df_ria[cn.CLS_FEATURE]
    for fit_result in fit_results:
      intersection = set(fit_result.sels).intersection(
          processed_features)
      if len(intersection) > 0:
        # Have already prcoessed this fit_result
        continue
      # Calculate for this FitResult
      df = self._calculateRIA(fit_result.sels,
          all_features)
      self.df_ria = pd.concat([self.df_ria, df])
      self._checkpoint()

  def _calculateRIA(self, cls_features,
      cmp_features):
    """
    Calculates RIA (relative incremental accuracy) for
    features used to train a classifier.
    :param list-object cls_features:
        set of features for classifier
    :param list-object cmp_features:
        features considered as alternatives 
        to the classifier features
    :return pd.DataFrame: columns
        cn.CLS_FEATURE - 
            feature removed from cls_features
        cn.CMP_FEATURE
            feature added from cmp_features
        cn.SCORE - score
    """
    ria_dct =  {
        cn.CLS_FEATURE: [],
        cn.CMP_FEATURE: [],
        cn.SCORE: [],
        }
    for ref_feature in cls_features:
      bcv_result =  \
          util_classifier.binaryCrossValidate(
          self._base_clf,
          self._df_X[cls_features], self._ser_y,
          partitions=self._partitions)
      score_with_ref =  bcv_result.score
      new_cls_features = list(cls_features)
      new_cls_features.remove(ref_feature)
      if len(new_cls_features) == 0:
        score_without_ref = NO_FEATURE_SCORE
      else:
        bcv_result =  \
            util_classifier.binaryCrossValidate(
            self._base_clf,
            self._df_X[new_cls_features],
            self._ser_y,
            partitions=self._partitions)
        score_without_ref =  bcv_result.score
      denom = score_with_ref - score_without_ref
      for cmp_feature in cmp_features:
        # Calculate the relative incremental accuracy
        new_cls_features.append(cmp_feature)
        bcv_result =  \
            util_classifier.binaryCrossValidate(
            self._base_clf,
            self._df_X[new_cls_features],
            self._ser_y,
            partitions=self._partitions)
        score_with_cmp = bcv_result.score
        new_cls_features.remove(cmp_feature)
        numr = score_with_cmp - score_without_ref
        if np.isclose(denom, 0):
          ria = np.nan
        else:
          ria = numr / denom
        ria_dct[cn.CLS_FEATURE].append(ref_feature)
        ria_dct[cn.CMP_FEATURE].append(cmp_feature)
        ria_dct[cn.SCORE].append(ria)
    return pd.DataFrame(ria_dct)
