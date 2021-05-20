""" Model for non-linear ODE """ 

import common_python.ODEModel.constants as cn
from common_python.sympy import sympyUtil as su

import numpy as np
import sympy

X = "x"
t = sympy.Symbol("t")
IS_TIMER = False


class ODEFixedPoint():

    def __init__(self, valueDct, jacobianMat, subs=None):
        if subs is None:
            subs = {}
        self.valueDct = su.evaluate(valueDct, subs=subs)  # state at fixed point
        self.stateSymVec = sympy.Matrix(list(valueDct.keys()))
        self.jacobianMat = su.evaluate(jacobianMat, subs=valueDct, isNumpy=False)
        self.jacobianMat = su.evaluate(self.jacobianMat, subs=subs,
              isNumpy=False)
        # Extract the eigenvalues
        self.eigenValues = list(self.jacobianMat.eigenvals().keys())
        self.eigenValues = list(self.jacobianMat.eigenvals().keys())
        self.eigenValues = [su.evaluate(e, subs=subs, isNumpy=False)
              for e in self.eigenValues]

    def getJacobian(self, subs=None):
        if subs is None:
            subs = {}
        return su.evaluate(self.jacobianMat, subs=subs)

    def getEigenvalues(self, subs=None):
        if subs is None:
            subs = {}
        return [su.evaluate(v, subs=subs) for v in self.eigenValuesLst]


class ODEModel():

    def __init__(self, stateDct, initialDct=None):
        """
        Parameters
        ----------
        stateDct: dict
            key: state Symbol
            value: expression
        initialDct: dict
            key: state Symbol
            value: initial values
        """
        self.stateDct = stateDct
        self.initialDct = initialDct
        self.stateSymVec = sympy.Matrix(list(stateDct.keys()))
        self.stateEprVec = sympy.Matrix([self.stateDct[s] for s in self.stateSymVec])
        self.jacobianMat = self.stateEprVec.jacobian(self.stateSymVec)
        self.fixedPoints = self._calcFixedPoints()


    def _calcFixedPoints(self):
        """
        Calculates the fixed points for the model.
        
        Returns
        -------
        list-FixedPoint
        """
        def mkFixedDcts(fixedPoints):
            def mkDct(points):
                if isinstance(points, dict):
                    return points
                return {v: p for p, v in zip(points, equationVars)}
            #
            if isinstance(fixedPoints, dict):
                fixedDcts = [mkDct(fixedPoints)]
            else:
                fixedDcts = []
                for fixedPoint in fixedPoints:
                    fixedDcts.append(mkDct(fixedPoint))
            return fixedDcts
        # Calculate the fixed points
        fixedPoints = sympy.solve(equationVec, equationVars)
        fixedDcts = mkFixedDcts(fixedPoints)
        return [FixedPoint(f, self.jacobianMat) for f in fixedDcts]
