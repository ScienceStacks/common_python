from common_python.LTIModel.LTISolver import LTISolver
import common_python.sympy.sympyUtil as su

import numpy as np
import pandas as pd
import sympy
import unittest


IGNORE_TEST = True
IS_PLOT = True
SYMBOLS = "X_0 X_1 x y z k0 k1 k2 k3 t"
su.addSymbols(SYMBOLS, dct=globals())
SUBS = {k0: 1, k1: 2, k2: 3, t: 100}
A_MAT = sympy.Matrix([ [   0,    0,          0,  0],
                       [   1,  -(k0 + k2),   0,  0], 
                       [   0,          k0, -k1,  0], 
                       [   0,          k2,  k1, -1],
                       ])
IMG_MAT = sympy.Matrix([ [1, 2], [-10, 1]])


#############################
# Tests
#############################
class TestLTISolver(unittest.TestCase):

    def setUp(self):
        self.init()
        aMat = sympy.Matrix([ [   0,    0,          0,  0],
                              [   1,  -(k0 + k2),   0,  0], 
                              [   0,          k0, -k1,  0], 
                              [   0,          k2,  k1, -1],
                            ])
        initialVec = sympy.Matrix([1, 0, 0, 0])
        rVec = sympy.Matrix([k0, 0, -k0])
        rVec = None
        self.model = LTISolver(aMat, initialVec, rVec=rVec)

    def init(self, aMat=A_MAT):
        initialVec = sympy.zeros(aMat.rows, 1)
        initialVec[0] = 1
        rVec = None
        self.model = LTISolver(aMat, initialVec, rVec=rVec)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.model.aMat.rows, 4)

    def testConstructor2(self):
        if IGNORE_TEST:
            return
        initialVec = sympy.zeros(IMG_MAT.rows)
        model = LTISolver(IMG_MAT, initialVec)
        eigenValTpl = su.asRealImag(model.eigenCollection.eigenInfos[0].val)
        self.assertTrue(np.abs(eigenValTpl[1]) > 0)

    def testSolve1(self):
        if IGNORE_TEST:
            return
        # Homogeneous equation with initial values
        subs = dict(SUBS)
        resultVec = self.model.solve(subs=subs)
        resultVec = resultVec.subs(subs)
        resultVec = sympy.simplify(resultVec)
        vals = [1, 0.25, 0.125, 1.0]
        for pos in range(len(resultVec)):
            num1 = float(resultVec[pos])
            num2 = float(vals[pos])
            self.assertTrue(np.isclose(num1, num2))

    def testSolve2(self):
        if IGNORE_TEST:
            return
        # Homogeneous equation with initial values
        self.init(aMat=IMG_MAT)
        resultVec = self.model.solve()
        # TODO: Need test

    def testEvaluate1(self):
        if IGNORE_TEST:
            return
        solutionVec = self.model.solve()
        resultVec = su.evaluate(solutionVec, subs=SUBS)
        vals = [1, 0.25, 0.125, 1.0]
        for pos in range(len(resultVec)):
            num1 = float(resultVec[pos][0])
            num2 = float(vals[pos])
            self.assertTrue(np.isclose(num1, num2))

    def testEvaluate2(self):
        # TESTING
        # Geometric multiplicity < algebraic multiplicity
        subs = dict(SUBS)
        subs[k1] = 2
        solutionVec = self.model.solve(subs=subs)
        resultVec = su.evaluate(solutionVec, subs=subs)
        vals = [1, 0.25, 0.125, 1.0]
        for pos in range(len(resultVec)):
            num1 = float(resultVec[pos][0])
            num2 = float(vals[pos])
            self.assertTrue(np.isclose(num1, num2))

    def testSolve3(self):
        if IGNORE_TEST:
            return
        aMat = sympy.Matrix([ [   0,           0,   0,  0], 
                               [   k0,  -(k1 + k3),  0,  0], 
                               [   0,          k1, -k2,  0], 
                               [   0,          k3,  k2, -k0],
                               ])
        initialVec = sympy.Matrix([1, 0, 0, 0])
        solver = LTISolver(aMat, initialVec)
        solver.plot(0, 5, 100, subs={k0:1, k1: 1, k2: 2, k3: 3})

    def testPlot(self):
        if IGNORE_TEST:
            return
        subs = dict(SUBS)
        del subs[t]
        self.model.plot(0, 10, 100, subs, isPlot=IS_PLOT, ylabel="values")



if __name__ == '__main__':
  unittest.main()
