import common_python.sympy.sympyUtil as su
import common_python.sympy.constants as cn

import numpy as np
import pandas as pd
import sympy
import unittest


IGNORE_TEST = False
IS_PLOT = False
VARIABLES = "X Y Z"


#############################
# Tests
#############################
class TestFunctions(unittest.TestCase):

    def setUp(self):
        su.addSymbols(VARIABLES, dct=globals())

    def testAddSymbols(self):
        if IGNORE_TEST:
            return
        names = ["xx", "yy"]
        su.addSymbols(" ".join(names))
        for name in names:
            self.assertTrue(name in locals().keys())
            expr = "isinstance(%s, sympy.Symbol)" % name
            self.assertTrue(eval(expr))

    def testAddSymbols2(self):
        if IGNORE_TEST:
            return
        names = ["xx", "yy"]
        su.addSymbols(" ".join(names), dct=globals())  # Ensure explicit opt works
        for name in names:
            self.assertTrue(name in globals().keys())
            expr = "isinstance(%s, sympy.Symbol)" % name
            self.assertTrue(eval(expr))

    def testRemoveSymbols(self):
        if IGNORE_TEST:
            return
        names = ["xx", "yy"]
        su.addSymbols(" ".join(names))
        su.removeSymbols(" ".join(names))
        for name in names:
            self.assertFalse(name in locals().keys())
            
    def testSubstitute(self):
        if IGNORE_TEST:
            return
        Y = su.substitute(2*X + 1, subs={X: Z})
        self.assertTrue("Z" in str(Y))

    def testEvaluate(self):
        if IGNORE_TEST:
            return
        val = su.evaluate(2*X + 1, subs={X: 2})
        self.assertEqual(val, 5)

    def testEvaluate2(self):
        if IGNORE_TEST:
            return
        expr = sympy.Matrix( [2*Z, Z**2])
        val = su.evaluate(expr, subs={Z: 2})
        self.assertEqual(expr.rows, np.shape(val)[0])
        self.assertEqual(expr.cols, np.shape(val)[1])

    def testMkVector(self):
        if IGNORE_TEST:
            return
        nameRoot = "z"
        numRow = 5
        vec = su.mkVector(nameRoot, numRow)
        newNames = [n for n in locals().keys()
              if (n[0] == nameRoot) and (len(n) == 3)]
        self.assertEqual(len(newNames), numRow)

    # FIXME:
    def testSolveLinearSystem(self):
        return
        if IGNORE_TEST:
            return
        data1 = [
              [0,  1, 0, 0, 1],
              [2,  -2, 0, 0, 0],
              [2,  -2, 0, 0, 0],
              [0,  0, 6, 6, 0]
              ]
        data2 = [
              [0,  1, 0, 0],
              [2,  -2, 0, 0],
              [2,  -2, 0, 0],
              [0,  0, 6, 6]
              ]
        bArr = np.array([1, 0, 0, 0])
        aArr = np.array(data2)
        result = np.linalg.solve(aArr, bArr)
        import pdb; pdb.set_trace()
        aMat = sympy.Matrix(data1)
        w, x, y, z = sympy.symbols("w x y z")
        result = sympy.solve_linear_system_LU(aMat, [w, x, y, z])
        import pdb; pdb.set_trace()
        x = aMat.LUsolve(bMat)
        result = su.solveLinearSystem(aMat, bMat)
        import pdb; pdb.set_trace()

    def testIsZero(self):
        if IGNORE_TEST:
            return
        self.assertTrue(su.isZero(0))
        self.assertFalse(su.isZero(1))
        self.assertTrue(su.isZero(X - X))

    def testIsVecZero(self):
        if IGNORE_TEST:
            return
        vec = sympy.Matrix([0, 0])
        self.assertTrue(su.isVecZero(vec))
        vec = sympy.Matrix([1, 0])
        self.assertFalse(su.isVecZero(vec))
        vec = sympy.Matrix([0, X - X])
        self.assertTrue(su.isVecZero(vec))
        vec = sympy.Matrix([Y, X - X])
        self.assertFalse(su.isVecZero(vec))

    def testExpressionToNumber(self):
        if IGNORE_TEST:
            return
        def test(cmplxVal, expected=None):
            val = su.expressionToNumber(cmplxVal)
            if expected is None:
                expected = cmplxVal
            self.assertTrue(expected == su.expressionToNumber(cmplxVal))
        #
        expression = 1e-9 * sympy.I
        test(expression.evalf(), expected=0)
        expression = 3 * sympy.I
        test(expression.evalf())
        expression = 3.0 + 1e-9 * sympy.I
        test(expression.evalf(), expected = 3.0)
        test(3.0)
        test(-3.0)
       


if __name__ == '__main__':
  unittest.main()
