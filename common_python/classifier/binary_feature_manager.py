'''Constructs features for a binary classifier.'''

"""
BinaryFeatureManager constructs the set of features
for a binary classifier. The algorithm has two
parts.
1. Forward selection. Choose features that increase
   the classification score up to a small difference
   from the score achieved using all features
   (best score).
   The parameters are:
     min_incr_score (minimum increase in score achieved
         by each feature)
     max_degrade (degradation from best score)
     max_iterations (maximum number of features
         considered)
2. Backwards elimination. Drop those features that
   don't significantly impact the score.
   The parameters are:
     min_incr_score (maximum amount by which the
         score can decrease if a feature is eliminated)
"""

from common_python.classifier import util_classifier
import common_python.constants as cn

import copy
import numpy as np
import pandas as pd
import random


# Default checkpoint callback
CHECKPOINT_CB = lambda : None
BINARY_CLASSES = [cn.NCLASS, cn.PCLASS]
MAX_BACKTRACK_ITERATIONS = 100


class BinaryFeatureManager(object):
  """
  Does feature selection for binary classes.
  Exposes the following instance variables
    1. score - score achieved for features
    2  best_score
    3. is_done - completed feature construction
    4. features selected for classifier
  This is a computationally intensive activity and so
  the implementation allows for restarts.
  """

  def __init__(self, df_X, ser_y, clf,
      checkpoint_cb=CHECKPOINT_CB,
      feature_selector=None,
      min_incr_score=MIN_INCR_SCORE,
      max_iter=MAX_ITER, 
      max_degrade=MAX_DEGRADE,
      ):
    """
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: binary class values (0, 1)
    :param Classifier clf:
        Exposes: fit, score, predict
    :param FeatureSelector feature_selector:
    :param float min_incr_score: min amount by which
        a feature must increase the score to be included
    :param int max_iter: maximum number of iterations
    :param float max_degrade: maximum difference between
        best score and actual
    """
    ########### PRIVATE ##########
    self._checkpoint_cb = checkpoint_cb
    self._clf = clf
    self._df_X = df_X
    self._ser_y = ser_y
    self._selector = feature_selector
    self._min_incr_score = min_incr_score
    self._max_iter = max_iter
    self._max_degrade = max_degrade
    self._test_idxs = self._makeTestIndices()
    ########### PUBLIC ##########
    # Score with all features
    self.best_score = util_classifier.scoreFeatures(
        clf, self._df_X, self._ser_y,
        test_idxs=self._test_idxs)
    # Score achieved for features in selector
    self.score = 0
    # Features found
    self.features = []

  @property
  def _completed_iterations(self):
    # Has begin running fit
    return len(self._selector.features)  \
        + len(self._selector.removes)

  @property
  def is_done(self):
    return len(self.features) > 0

  def _makeTestIndices(self):
    """
    Constructs the test indices so that positive
    and negative classes are equally represented.
    :return list-object:
    Notes
      1. Assumes that number of PCLASS < NCLASS
    """
    pclass_idxs = self._ser_y[ser_ser_y==cn.PCLASS].index
    nclass_idxs = self._ser_y[ser_ser_y==cn.NCLASS].index
    # Sample w/o replacement from the larger set
    if len(pclass_idxs) < len(nclass_idxs):
      length = len(pclass_idxs)
      test_idxs = pclass_idxs
      sample_idxs = nclass_idxs
    else:
      length = len(nclass_idxs)
      test_idxs = nclass_idxs
      sample_idxs = pclass_idxs
    sample_idxs = random.sample(sample_idxs, length)
    test_idxs.extend(sample_idxs)
    return test_idxs

  def run(self):
    """
    Construct the features, handling restarts by saving
    state and checkpointing.
    Result is in self.features.
    """
    # Forward selection of features
    for _ in range(len(self._selector.getCandidates())):
      if self._completed_iterations  \
          % CHECKPOINT_INTERVAL == 0:
        self._checkpoint_cb()  #  Save state
      if self._completed_iterations >= self._max_iter:
        break  # Reached maximum number of iterations
      if not self.selector.add():
        break  # No more features to add
      new_score = util_classifier.scoreFeatures(
          self._clf, self._df_X, self._ser_y,
          features=self._selector.features,
          test_idxs=self._test_idxs)
      if new_score - self.score > self._min_incr_score:
          # Feature is acceptable
          self.score = new_score
      else:
        # Remove the feature
        self.selector.remove(cls)
      # See if close enough to best possible score
      if self.best_score - self.score  \
          < self._max_degrade:
        break
    # Backwards elimination to delete unneeded feaures
    # Eliminate features that do not affect accuracy
    for _ in range(MAX_BACKTRACK_ITERATIONS):
      is_done = True
      for feature in self.selector.features:
        # Temporarily delete the feature
        self.selector.remove(feature=feature)
        new_score = util_classifier.scoreFeatures(
            self._clf, self._df_X, self._ser_y,
            features=self._selector.features,
            test_idxs=self._test_idxs)
        if self.score - new_score > self._min_incr_score:
          # Restore the feature
          self.selector.add(cls, feature=feature)
        else:
          # Permanently delete the feature
          self.score = new_score
          is_done = False
      if is_done:
        break
    #
    self.features = self._selector.features
