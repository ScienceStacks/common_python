import common_python.sympy.sympyUtil as su
import common_python.ODEModel.constants as cn
from common_python.ODEModel.ODEModel import ODEModel, ODEFixedPoint

import numpy as np
import pandas as pd
import sympy
import unittest


IGNORE_TEST = False
IS_PLOT = False
VARIABLES = "t A P M K_AN K_AM N Kd_A Kd_M K_MP K_PA k_0 K_MP kd_M"
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


#############################
# Tests
#############################
class TestODEFixedPoint(unittest.TestCase):

    def setUp(self):
        self.init(STATE_DCT, VALUE_DCT)

    def init(self, stateDct, valueDct, subs=SUBS):
        self.stateSymVec = sympy.Matrix(list(stateDct.keys()))
        self.stateSymVec = sympy.Matrix(list(stateDct.keys()))
        self.stateEprVec = sympy.Matrix([stateDct[s]
              for s in self.stateSymVec])
        self.jacobianMat  = self.stateEprVec.jacobian(self.stateSymVec)
        self.fixedPoint = ODEFixedPoint(valueDct, self.jacobianMat, subs=subs)

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
        self.init(STATE2_DCT, VALUE_DCT)
        self.assertTrue("Matrix" in str(type(self.fixedPoint.jacobianMat)))
        numReal = sum([v.is_real for v in self.fixedPoint.eigenValues])
        self.assertEqual(numReal, 1)


class TestODEModel(unittest.TestCase):

    def setUp(self):
        su.addSymbols(VARIABLES, dct=globals())
        self.eigenInfo = mkEigenInfo()



if __name__ == '__main__':
  unittest.main()
