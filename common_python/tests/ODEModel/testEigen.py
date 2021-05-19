import common_python.sympy.sympyUtil as su
import common_python.ODEModel.constants as cn
from common_python.ODEModel.eigenCollection import EigenCollection

import numpy as np
import pandas as pd
import sympy
import unittest


IGNORE_TEST = False
IS_PLOT = False
VARIABLES = "t X Y Z k0 k1 k2"
su.addSymbols(VARIABLES, dct=globals())
SUBS = {k0: 1, k1: 2, k2: 3, t: 100}
A_MAT = sympy.Matrix([ [   0,    0,          0,  0],
                       [   1,  -(k0 + k2),   0,  0], 
                       [   0,          k0, -k1,  0], 
                       [   0,          k2,  k1, -1],
                       ])
# No. eigenvalues: 2, Algebraic multiplicity: 1, Geometric multiplicity: 1
FULL_MAT = sympy.Matrix([
      [1, 2],
      [2, 1],
      ])
# No. eigenvalues: 1, Algebraic multiplicity: 2, Geometric multiplicity: 1
DEFICIENT_MAT = sympy.Matrix([
      [1, 0],
      [2, 1],
      ])
LARGE_MAT = sympy.Matrix([
      [1, 0, 2],
      [2, 1, 3],
      [1, 3, 3],
      ])
SUBS = {k0: 1, k1: 2, k2: 3}
IMAG_MAT = sympy.Matrix([ [1, 2], [-10, 1]])

############ FUNCTIONS #############
def mkEigenInfo(mat=FULL_MAT):
    aMat = mat.copy()
    eigenCollection = EigenCollection(aMat)
    return eigenCollection.eigenInfos[0].copy()


#############################
# Tests
#############################
class TestEigenInfo(unittest.TestCase):

    def setUp(self):
        su.addSymbols(VARIABLES, dct=globals())
        self.eigenInfo = mkEigenInfo()

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.eigenInfo.mul, 1)

    def testCompleteEigenvectors(self):
        if IGNORE_TEST:
            return
        def test(numvec):
            eigenInfo = mkEigenInfo(mat=LARGE_MAT)
            curSize = len(eigenInfo.vecs)
            eigenInfo.mul = numvec
            eigenInfo.completeEigenvectors()
            self.assertEqual(numvec, len(eigenInfo.vecs))
            return eigenInfo
        #
        _ = test(1)
        _ = test(2)

    def testComplexEigenvalues(self):
        if IGNORE_TEST:
            return
        self.eigenInfo = mkEigenInfo(mat=IMAG_MAT)
        self.assertTrue(np.isclose(1-4.472135949j, self.eigenInfo.val))


class TestEigenCollection(unittest.TestCase):

    def setUp(self):
        su.addSymbols(VARIABLES, dct=globals())
        self.aMat = FULL_MAT.copy()
        self.eigenCollection = EigenCollection(self.aMat)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(len(self.eigenCollection.eigenInfos), 2)
        self.assertEqual(self.eigenCollection.eigenInfos[0].val, -1.0)
        self.assertEqual(self.eigenCollection.eigenInfos[1].val, 3.0)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        eigen = EigenCollection(DEFICIENT_MAT)
        self.assertEqual(len(eigen.eigenInfos), 1)
        eigenInfo = eigen.eigenInfos[0]
        self.assertEqual(eigenInfo.val, 1.0)
        self.assertEqual(eigenInfo.mul, 2)

    def testComplexEigenvalues(self):
        if IGNORE_TEST:
            return
        eigenCollection = EigenCollection(IMAG_MAT)
        eigenVec1= eigenCollection.eigenInfos[0].vecs[0]
        eigenVec2= eigenCollection.eigenInfos[1].vecs[0]
        vec = eigenVec1 + eigenVec2
        self.assertEqual(vec[0] , 0)
        self.assertEqual(vec[1], 2 * eigenVec1[1])

    def testPruneConjugates(self):
        if IGNORE_TEST:
            return
        eigenCollection = EigenCollection(IMAG_MAT)
        eigenInfos = eigenCollection.pruneConjugates()
        self.assertEqual(len(eigenInfos), 1)
        self.assertEqual(len(eigenCollection.eigenInfos), 2)


if __name__ == '__main__':
  unittest.main()
