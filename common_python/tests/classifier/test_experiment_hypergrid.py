from common_python.classifier.experiment_hypergrid  \
    import ExperimentHypergrid
from common_python.testing import helpers
import common_python.constants as cn
from common_python.tests.classifier import helpers as test_helpers

import pandas as pd
import numpy as np
from sklearn import svm
import unittest

IGNORE_TEST = False
IS_PLOT = True


class TestExperimentHypergrid(unittest.TestCase):

  def setUp(self):
    self.experiment = ExperimentHypergrid()

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(len(self.experiment.grid), 2)
    self.assertLess(len(self.experiment.pos_vecs)
        + len(self.experiment.neg_vecs),
        np.size(self.experiment.grid) / 2)

  def testPlotGrid(self):
    # Smoke test
    self.experiment.plotGrid(is_plot=IS_PLOT)

  


if __name__ == '__main__':
  unittest.main()
