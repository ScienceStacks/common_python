import common_python.sympy.sympyUtil as su
import common_python.ODEModel.constants as cn
from common_python.ODEModel.ODEModel import ODEModel, FixedPoint, EigenEntry

import itertools
import matplotlib
import numpy as np
import os
import pandas as pd
import sympy
import tellurium as te
import unittest


IGNORE_TEST = True
IS_PLOT = True
VARIABLES = "t A P M K_AN K_AM N Kd_A Kd_M K_MP K_PA k_0 K_MP kd_M X Y"
su.addSymbols(VARIABLES, dct=globals())
STATE_DCT = {
      A: K_AN * N - Kd_A * A * M,
      P: K_PA * A - k_0,
      M: K_MP * P - Kd_M * M,
      }
STATE2_DCT = {
      A: K_AN * N - Kd_A*A + K_AM * M,
      P: -K_PA * A * M  + k_0,
      M: K_MP * P - Kd_M * M,
      }
STATE_SYM_VEC = sympy.Matrix(list(STATE_DCT.keys()))
STATE_EPR_VEC = sympy.Matrix([STATE_DCT[s] for s in STATE_SYM_VEC])
JACOBIAN_MAT = STATE_EPR_VEC.jacobian(STATE_SYM_VEC)
VALUE_DCT = {
      A: k_0/K_PA, 
      P: K_AN*K_PA*Kd_M*N/(K_MP*Kd_A*k_0),
      M: K_AN*K_PA*N/(Kd_A*k_0)
      }
SUBS = {Kd_A: 1, K_MP: 1, K_PA: 1, k_0: 2.0, Kd_M: 1, N: 1,
      K_AN: 1, Kd_M: 1, Kd_A: 1, K_AM: 1}
ANTIMONY_FILE = "antimony.ant"
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ANTIMONY_FILE = os.path.join(TEST_DIR, "antimony.ant")


#############################
# Tests
#############################
class TestEigenEntry(unittest.TestCase):

    def setUp(self):
        self.init()

    def init(self, value=X, mul=1, vectors=None):
        self.value = value
        self.mul = mul
        if vectors is None:
            vectors = [sympy.Matrix([X, 2 * X])]
        self.vectors = vectors
        self.entry = EigenEntry(self.value, self.mul, self.vectors)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(self.entry.isReal)

    def testConstructor2(self):
        if IGNORE_TEST:
            return
        self.init(value= (2+3j))
        self.assertFalse(self.entry.isReal)

    def testGetEigenvalues(self):
        if IGNORE_TEST:
            return
        # Complex
        self.init(value= (2+3j))
        values = self.entry.getEigenvalues()
        self.assertEqual(len(values), 2)
        self.assertTrue(su.isConjugate(values[0], values[1]))
        # Symbol
        self.init(value=1*X)
        values = self.entry.getEigenvalues(subs={X: 1})
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], 1)

    def testGetEigenvectors(self):
        if IGNORE_TEST:
            return
        vectors = self.entry.getEigenvectors()
        self.assertEqual(len(vectors), 1)

    def testEquals(self):
        if IGNORE_TEST:
            return
        self.assertTrue(self.entry.equals(self.entry))
        entry = EigenEntry(self.entry.value+1,
            self.entry.algebraicMultiplicity,
            self.vectors)
        self.assertFalse(entry.equals(self.entry))



class TestFixedPoint(unittest.TestCase):

    def setUp(self):
        self.init(STATE_DCT, VALUE_DCT)

    def init(self, stateDct, valueDct, subs=SUBS, isEigenvecs=True):
        self.stateSymVec = sympy.Matrix(list(stateDct.keys()))
        self.stateSymVec = sympy.Matrix(list(stateDct.keys()))
        self.stateEprVec = sympy.Matrix([stateDct[s] for s in self.stateSymVec])
        self.jacobianMat  = self.stateEprVec.jacobian(self.stateSymVec)
        self.fixedPoint = FixedPoint(valueDct, self.jacobianMat, subs=subs,
              isEigenvecs=isEigenvecs)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue("Matrix" in str(type(self.fixedPoint.jacobianMat)))
        values = range(JACOBIAN_MAT.rows)
        positions = [x for x in itertools.product(values, values)]
        positions = set(positions).difference([(0, 0), (0, 1)])
        jacobianMat = su.evaluate(JACOBIAN_MAT, subs=VALUE_DCT, isNumpy=False)
        jacobianMat = su.evaluate(jacobianMat, subs=SUBS, isNumpy=False)
        self.assertTrue(jacobianMat == self.fixedPoint.jacobianMat)

    def testConstructor2(self):
        if IGNORE_TEST:
            return
        self.init(STATE2_DCT, VALUE_DCT, subs=SUBS, isEigenvecs=False)
        self.assertTrue("Matrix" in str(type(self.fixedPoint.jacobianMat)))
        numReal = sum([1 for e in self.fixedPoint.eigenEntries
              if np.abs(np.imag(e.value)) < 0.001])
        self.assertEqual(numReal, 1)

    def testGetJacobian(self):
        if IGNORE_TEST:
            return
        mat = self.fixedPoint.getJacobian()
        self.assertTrue("Matrix" in str(type(mat)))

    def testGetEigenvalues(self):
        if IGNORE_TEST:
            return
        eigenvalues = self.fixedPoint.getEigenvalues()
        self.assertEqual(len(eigenvalues), 3)

    def testEquals(self):
        if IGNORE_TEST:
            return
        self.assertTrue(self.fixedPoint.equals(self.fixedPoint))
        fixedPoint = self.fixedPoint.copy()
        sym = list(fixedPoint.valueDct.keys())[0]
        fixedPoint.valueDct[sym] = 1
        self.assertFalse(fixedPoint.equals(self.fixedPoint))


class TestODEModel(unittest.TestCase):

    def setUp(self):
        self.model = ODEModel(STATE_DCT, isEigenvecs=False)

    def testConstrutor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(len(self.model.fixedPoints), 1)
        self.assertGreater(len(self.model.jacobianMat.free_symbols), 0)
   
    def testGetFixedPointValues(self):
        if IGNORE_TEST:
            return
        fixedPointValues = self.model.getFixedPointValues(subs=SUBS)
        self.assertEqual(len(fixedPointValues), 1)
        fixedPointValue = fixedPointValues[0]
        diff = set(fixedPointValue.keys()).symmetric_difference(STATE_SYM_VEC)
        self.assertEqual(len(diff), 0)
        trues = [isinstance(su.expressionToNumber(v), float)
              for v in fixedPointValue.values()]
        self.assertTrue(all(trues))

    def testBug1(self):
        if IGNORE_TEST:
            return
        modelA = ODEModel(STATE_DCT, isEigenvecs=False)
        fps = modelA.getFixedPointValues()
        for fp in modelA.fixedPoints:
            eigenvalues = fp.getEigenvalues(subs=SUBS)
        self.assertEqual(len(eigenvalues), 3)

    def testCalcFixedPoints(self):
        if IGNORE_TEST:
            return
        symbols = "A K_AN K_AM N K_PA K_MP rd_A"
        symbols += " Kd_A Kd_M Kd_P K_Mp A M"
        symbols += " P k_0 M rd_M"
        su.addSymbols(symbols, dct=globals())
        stateDct = {
             A: K_AN * N - Kd_A * A * M,
             P: K_PA * A - k_0,
             M: K_MP * P - Kd_M * M,
        }
        subs = {k_0: 0.3, K_PA: 2, K_AN: 0.5}
        model = ODEModel(stateDct)
        fixedPointCopy = model.fixedPoints[0].copy(subs=subs)
        newFixedPoint = self.model._calcFixedPoints(subs=subs)[0]
        self.assertTrue(fixedPointCopy.equals(newFixedPoint))

    def testMkODEModel(self):
        if IGNORE_TEST:
            return
        rr = te.loada(ANTIMONY_FILE)
        modelInfo = ODEModel.mkODEModel(rr, isEigenvecs=False, isFixedPoints=False)
        self.assertTrue(isinstance(modelInfo.mdl, ODEModel))
        
    def testPlotJacobian(self):
        # TESTING
        subs = dict(SUBS)
        subs[M] = 0.5
        subs[A] = 2
        self.model.plotJacobian(isPlot=IS_PLOT, subs=subs)


if __name__ == '__main__':
  if IS_PLOT:
    matplotlib.use('TkAgg')
  unittest.main()
