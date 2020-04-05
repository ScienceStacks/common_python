from common_python.classifier import util_classifier
from common_python.tests.classifier  \
    import helpers as test_helpers

import pandas as pd
import numpy as np
import unittest


IGNORE_TEST = False


class TestFunctions(unittest.TestCase):
 
  def testFindAdjacentStates(self):
    SER_Y = pd.Series({
        "0-1": 0,
        "0-2": 0,
        "1-1": 1,
        "1-2": 1,
        "2-1": 2,
        "2-2": 2,
        "2-3": 2,
        })
    adjacents = util_classifier.findAdjacentStates(
        SER_Y, "0-1")
    self.assertTrue(np.isnan(adjacents.prv))
    self.assertTrue(np.isnan(adjacents.p_dist))
    self.assertEqual(adjacents.nxt, 1)
    self.assertEqual(adjacents.n_dist, 2)
    #
    adjacents = util_classifier.findAdjacentStates(
        SER_Y, "2-3")
    self.assertTrue(np.isnan(adjacents.nxt))
    self.assertTrue(np.isnan(adjacents.n_dist))
    self.assertEqual(adjacents.prv, 1)
    self.assertEqual(adjacents.p_dist, 3)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
  unittest.main()
