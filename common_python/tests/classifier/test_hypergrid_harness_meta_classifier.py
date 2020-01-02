from common_python.classifier.hypergrid_harness_meta_classifier  \
    import HypergridHarnessMetaClassifier
from common_python.classifier.meta_classifier  \
    import MetaClassifierDefault, MetaClassifierPlurality,  \
    MetaClassifierAugment, MetaClassifierAverage, \
    MetaClassifierEnsemble
from common_python.testing import helpers
import common_python.constants as cn

import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = False
IS_PLOT = False
NUM_DIM = 2
DENSITY = 4
OFFSET = 0
IDX_PLURALITY = 0
IDX_DEFAULT = 1
IDX_AUGMENT = 2
IDX_AVERAGE = 3
IDX_ENSEMBLE = 4


class TestHypergridHarnessMetaClassifier(unittest.TestCase):

  def setUp(self):
    mclfs = [
        MetaClassifierPlurality(),  # IDX_PLURALITY
        MetaClassifierDefault(),    # IDX_DEFAULT
        MetaClassifierAugment(),    # IDX_AUGMENT
        MetaClassifierAverage(),    # IDX_AVERAGE
        MetaClassifierEnsemble(),   # IDX_ENSEMBLE
        ]
    self.harness = HypergridHarnessMetaClassifier(
        mclfs, density=DENSITY)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(self.harness._density, DENSITY)
    self.assertGreater(len(self.harness.grid), 0)


if __name__ == '__main__':
  unittest.main()
