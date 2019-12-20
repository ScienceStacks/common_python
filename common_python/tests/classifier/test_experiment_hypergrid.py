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


class TestExperimentHypergrid(unittest.TestCase):

  def setUp(self):
    self.experiment = ExperimentHypergrid()

  def testConstructor(self):
    self.assertEqual(len(self.experiment.grid), 2)
    import pdb; pdb.set_trace()
  


if __name__ == '__main__':
  unittest.main()
