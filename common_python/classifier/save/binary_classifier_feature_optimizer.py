'Constructs features for a binary classifier.'''

"""
BinaryFeatureClassifierOptimizer selecteds a
set of features that optimize a binary classifier.
The software constraints of the optimization are:
  (a) Minimize the number of features
  (b) The features selected result in a classifier
      with a desired accuracy (desired_accuracy)
The hard constraints are:
  (i)  Maximum number of features considered (max_iter)
  (ii) Minimum increase in accuracy when including the
      feature (min_incr_score)

The algorithm operates in two phases.
1. Forward selection. Choose features that increase
   the classification score up to a small difference
   from the score achieved using all features
   (best score).
   The parameters are:
     min_incr_score (minimum increase in score achieved
         by each feature)
     desired_accuracy (degradation from best score)
     max_iterations (maximum number of features
         considered)
2. Backwards elimination. Drop those features that
   don't significantly impact the score.
   The parameters are:
     min_incr_score (maximum amount by which the
         score can decrease if a feature is eliminated)

Usage: Fits are one-shot. Create a new instance
for each fit.
  optimizer = BinaryClassifierFeatureOptimizer()
  optimizer.fit(df_X, ser_y)
  features = optimizer.features
"""

from common_python.classifier import util_classifier
from common_python.classifier.feature_collection  \
    import FeatureCollection
import common_python.constants as cn

import copy
import numpy as np
import pandas as pd
import random
from sklearn import svm


# Default checkpoint callback
CHECKPOINT_CB = lambda : None
CHECKPOINT_INTERVAL = 100
BINARY_CLASSES = [cn.NCLASS, cn.PCLASS]
MAX_ITER = 100
MAX_BACKWARD_ITER = MAX_ITER  # Max
MIN_INCR_SCORE = 0.01
DESIRED_ACCURACY = 1.0
NUM_HOLDOUTS = 1  # Holdouts in cross validation
NUM_CROSS_ITER = 20  # Cross validation iterations


class BinaryClassifierFeatureOptimizer(object):
  """
  Does feature selection for binary classes.
  Exposes the following instance variables
    1. score - score achieved for features
    2  all_score
    3. features selected for classifier
    4. is_done - completed processing
  This is a computationally intensive activity and so
  the implementation allows for restarts.
  """

  def __init__(self, base_clf=svm.LinearSVC(),
      checkpoint_cb=CHECKPOINT_CB,
      feature_collection=None,
      min_incr_score=MIN_INCR_SCORE,
      max_iter=MAX_ITER, 
      desired_accuracy=DESIRED_ACCURACY,
      num_holdouts=NUM_HOLDOUTS,
      num_cross_iter=NUM_CROSS_ITER
      ):
    """
    :param Classifier base_clf:
        Exposes: fit, score, predict
    :param FeatureCollection feature_collection:
    :param float min_incr_score: min amount by which
        a feature must increase the score to be included
    :param int max_iter: maximum number of iterations
    :param float desired_accuracy: maximum difference between
        best score and actual
    :param int num_holdouts: holdouts in cross validation
    :param int num_cross_iter: number of iterations
        in cross validation
    """
    ########### PRIVATE ##########
    self._base_clf = copy.deepcopy(base_clf)
    self._checkpoint_cb = checkpoint_cb
    self._iteration = -1  # Counts feature evaluations
    self._collection = feature_collection
    self._min_incr_score = min_incr_score
    self._max_iter = max_iter
    self._desired_accuracy = desired_accuracy
    self._num_holdouts = num_holdouts
    self._num_cross_iter = num_cross_iter
    self._partitions = None  # list of train, test data
    ########### PUBLIC ##########
    # Score with all features
    self.all_score = None  # Assigned in fit
    # Score achieved for features in collection
    self.score = 0
    # Collection of features selected for classifier
    self.selecteds = []
    # Flag to indicate completed processing
    self.is_done = False

  @property
  def num_iteration(self):
    return self._iteration

  def _updateIteration(self):
    if self._iteration % CHECKPOINT_INTERVAL == 0:
      self._checkpoint_cb()  #  Save state
    self._iteration += 1

  def _evaluate(self, df_X, ser_y, features=None):
    """
    Constructs cross validated features.
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: binary class values (0, 1)
    :return float: score
    """
    if self._partitions is None:
      self._partitions =  \
          [util_classifier.partitionByState(
          ser_y, holdouts=self._num_holdouts)
          for _ in range(self._num_cross_iter)]
    if features is None:
      features = self._collection.chosens
    scores = []
    for train_idxs, test_idxs in self._partitions:
      scores.append(util_classifier.scoreFeatures(
          self._base_clf, df_X, ser_y,
          features=features,
          train_idxs=train_idxs, test_idxs=test_idxs))
    return np.mean(scores)

  def fit(self, df_X, ser_y):
    """
    Construct the features, handling restarts by saving
    state and checkpointing.
    :param pd.DataFrame df_X:
        columns: features
        index: instances
    :param pd.Series ser_y:
        index: instances
        values: binary class values (0, 1)
    """
    if len(ser_y.unique()) != 2:
      raise ValueError("Must have exactly two classes.")
    df_X = copy.deepcopy(df_X)
    ser_y = copy.deepcopy(ser_y)
    df_X = df_X.sort_index()
    ser_y = ser_y.sort_index()
    # Initialization
    if self._collection is None:
      self._collection = FeatureCollection(df_X, ser_y)
    self.all_score = self._evaluate(df_X, ser_y,
        features=df_X.columns.tolist())
    # Forward selection of features
    for _ in range(len(self._collection.getCandidates())):
      self._updateIteration()  # Handles checkpoint
      if self._iteration >= self._max_iter:
        break  # Reached maximum number of iterations
      if not self._collection.add():
        break  # No more features to add
      new_score = self._evaluate(df_X, ser_y)
      if new_score - self.score > self._min_incr_score:
          # Feature is acceptable
          self.score = new_score
      else:
        # Remove the feature
        self._collection.remove()
      # See if close enough to best possible score
      if self.score >= self._desired_accuracy:
        break
    # Backwards elimination to delete unneeded feaures
    # Eliminate features that do not affect accuracy
    for _ in range(MAX_BACKWARD_ITER):
      is_changed = False
      self._updateIteration()
      for feature in self._collection.chosens:
        if len(self._collection.chosens) == 1:
          break
        # Temporarily delete the feature
        self._collection.remove(feature=feature)
        new_score = self._evaluate(df_X, ser_y)
        if self.score - new_score > self._min_incr_score:
          # Restore the feature
          self._collection.add(feature=feature)
        else:
          # Permanently delete the feature
          self.score = new_score
          is_changed = True
      if not is_changed:
        break
    #
    self.selecteds = list(self._collection.chosens)
    self._checkpoint_cb()
    is_done = True
