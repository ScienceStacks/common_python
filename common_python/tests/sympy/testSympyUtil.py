import common_python.sympy.sympyUtil as su
import common_python.sympy.constants as cn

import numpy as np
import pandas as pd
import sympy
import unittest


IGNORE_TEST = False
IS_PLOT = False
VARIABLES = "t X Y Z k0 k1 k2"
su.addSymbols(VARIABLES, dct=globals())


#############################
# Tests
#############################
class TestFunctions(unittest.TestCase):

    def setUp(self):
        pass

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
        X, Y, Z = sympy.symbols("X Y Z")
        expr = 2*X + 1
        subs = {X: Z}
        Y = su.substitute(expr, subs=subs)
        self.assertTrue("Z" in str(Y))
        # Test for substitution if duplicate name but different symbol
        expr = 2*X + 1
        X = sympy.Symbol("X")
        Y = su.substitute(expr, subs={X: Z})

    def testEvaluate(self):
        if IGNORE_TEST:
            return
        val = su.evaluate(2*X + 1, subs={X: 2})
        self.assertEqual(val, 5)
        #
        val = su.evaluate(2*X + 1, subs={X: 2})
        self.assertEqual(val, 5)
        #
        val = su.evaluate(2*X + 1, subs={X: 2}, isNumpy=False)
        self.assertTrue(val.is_number)

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

    def testRecursiveEquals(self):
        if IGNORE_TEST:
            return
        self.assertTrue(su.recursiveEquals(3.0, 3.0))
        expr = X * 3.0
        self.assertTrue(su.recursiveEquals(expr, 3.0, subs={X: 1.0}))
        # Not equal
        self.assertFalse(su.recursiveEquals(expr, 3.0, subs={X: 2.0}))
        # Equal structures
        result = su.recursiveEquals((expr, expr), (3.0, 3.0), subs={X: 1.0})
        self.assertTrue(result)
        obj1 = sympy.Matrix([expr, expr])
        obj2 = sympy.Matrix([3.0, 3.0])
        result = su.recursiveEquals(obj1, obj2, subs={X: 1.0})
        self.assertTrue(result)
        # Complex structures
        obj1 = sympy.Matrix([expr, expr])
        obj1 = sympy.Matrix([obj1, obj1, obj1])
        obj2 = sympy.Matrix([3.0, 3.0])
        obj2 = sympy.Matrix([obj2, obj2, obj2])
        result = su.recursiveEquals(obj1, obj2, subs={X: 1.0})
        self.assertTrue(result)
        result = su.recursiveEquals(obj1, obj2, subs={X: 2.0})
        self.assertFalse(result)
        # Different structures
        result = su.recursiveEquals(expr, (3.0, 3.0), subs={X: 1.0})
        self.assertFalse(result)
        # Same structures, different values
        result = su.recursiveEquals((expr, expr), (3.0, 3.0), subs={X: 1.1})
        self.assertFalse(result)

    def testRecursiveEvaluate(self):
        if IGNORE_TEST:
            return
        def test(obj, expected, subs={}):
            newObj = su.recursiveEvaluate(obj, subs=subs)
            self.assertTrue(su.recursiveEquals(newObj, obj, subs=subs))
        # Tests
        expr = 3 * X
        # Complex structures
        obj1 = sympy.Matrix([expr, expr])
        obj1 = sympy.Matrix([obj1, obj1, obj1])
        obj2 = sympy.Matrix([3.0, 3.0])
        obj2 = sympy.Matrix([obj2, obj2, obj2])
        test(obj1, obj2, subs={X: 1.0})
        # Simple tests
        test(3.0, 3.0)
        test(expr, 3.0, subs={X: 1})

    def testIsNumber(self):
        if IGNORE_TEST:
            return
        def test(obj, isNum):
            self.assertEqual(su.isNumber(obj), isNum)
        #
        test(3.0, True)
        test(complex(3.0), True)
        test(X, False)
        expr = 3 * X
        expr = expr.subs(X, 1)
        test(expr, True)

    def testIsSympy(self):
        if IGNORE_TEST:
            return
        self.assertTrue(su.isSympy(X))
        self.assertFalse(su.isSympy(3.0))
        self.assertTrue(su.isSympy(sympy.Matrix([X])))
        expr = 3 * X
        expr = expr.subs(X, 1)
        self.assertTrue(su.isSympy(expr))

    def testIsConjugate(self):
        if IGNORE_TEST:
            return
        cmplx1 = 2 + 3j
        cmplx2 = 2 - 3j
        self.assertTrue(su.isConjugate(cmplx1, cmplx2))
        expr = X * cmplx1
        expr = expr.subs(X, 1.0)
        self.assertTrue(su.isConjugate(expr, cmplx2))

    def testVectorAsRealImag(self):
        if IGNORE_TEST:
            return
        vec = sympy.Matrix([ 2 + 3j, 2j, 2])
        realVec, imagVec = su.vectorAsRealImag(vec)
        for idx, item in enumerate(vec):
            real, imag = su.asRealImag(item)
            self.assertEqual(real, realVec[idx])
            self.assertEqual(imag, imagVec[idx])

    def testHasSymbols(self):
        if IGNORE_TEST:
            return
        self.assertTrue(su.hasSymbols(X))
        self.assertFalse(su.hasSymbols(3))
        self.assertTrue(su.hasSymbols([X, 3]))
        self.assertFalse(su.hasSymbols([3, 3]))

    def testGetDistinctSymbols(self):
        if IGNORE_TEST:
            return
        dct = {
            X: 3 * Y,
            Y: 3 * Z,
            Z: 4 + Y,
            }
        distinctSymbols = su.getDistinctSymbols(dct)
        diff = set(distinctSymbols).symmetric_difference([Y, Z])
        self.assertEqual(len(diff), 0)
        #
        symbols = [Y]
        distinctSymbols = su.getDistinctSymbols(dct, symbols=symbols)
        self.assertEqual(distinctSymbols[0], Y)
        self.assertEqual(len(distinctSymbols), 1)

    def testSolveLinearSingular(self):
        if IGNORE_TEST:
            return
        aMat = sympy.Matrix([
              [1, 0, 1],
              [1, 1, 0],
              ])
        bVec = sympy.zeros(rows=aMat.rows, cols=1)
        result = su.solveLinearSingular(aMat,bVec, isParameterized=True)
        sym = result.free_symbols
        newResult = aMat.multiply(result.subs(sym, 1))
        self.assertTrue(newResult == bVec)


if __name__ == '__main__':
  unittest.main()
