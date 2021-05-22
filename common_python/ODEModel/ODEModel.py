""" 
Model for non-linear ODE.

An ODEModel has one or more FixedPoint.
Each FixedPoint has a Jacobian and one or more EigenEntry.
An EigenEntry has 1 (if real) or 2 (if complex) eigenvalues
and one or more eigenvectors.
 """ 

import common_python.ODEModel.constants as cn
from common_python.sympy import sympyUtil as su

import numpy as np
import sympy

X = "x"
t = sympy.Symbol("t")
IS_TIMER = False


class EigenEntry():
    """ Contains all information related to an eigenvalue """

    def __init__(self, value, algebraicMultiplicity, vectors):
        """
        Complex conjugates have a single entry.

        Parameters
        ----------
        value: number (Eigenvalue)
        algebraicMultiplicity: int
            algebraic multiplicity
        vectors: list-sympy.Matrix N X 1 (or None)
        """
        self.value = value
        self.algebraicMultiplicity = algebraicMultiplicity
        self.vectors = vectors
        self.isReal = su.isReal(self.value)

    def getEigenvalues(self, subs=None):
        """
        Parameters
        ----------
        subs: dict
        
        Returns
        -------
        list-number
        """
        if subs is None:
            subs = {}
        values = [su.evaluate(self.value, subs=subs)]
        if not self.isReal:
            real, imag = su.asRealImag(values[0])
            values = [values[0], real - imag*1j]
        return values

    def getEigenvectors(self, subs=None):
        if self.vectors is None:
            return None
        if subs is None:
            subs = {}
        return [su.evaluate(v, subs=subs) for v in self.vectors]


class FixedPoint():

    def __init__(self, valueDct, jacobianMat, subs=None, isEigenvecs=True):
        """
        Parameters
        ----------
        valuedDct: dict
            key: state variable
            value: number/expression
        jacobianMat: sympy.Matrix N X N
        subs: dict
        isEigenvecs: bool
            include eigenvectors
        """
        if subs is None:
            subs = {}
        self.valueDct = su.evaluate(valueDct, subs=subs)  # state at fixed point
        self.stateSymVec = sympy.Matrix(list(valueDct.keys()))
        self.jacobianMat = su.evaluate(jacobianMat, subs=valueDct, isNumpy=False)
        self.jacobianMat = su.evaluate(self.jacobianMat, subs=subs, isNumpy=False)
        # Create the EigenEntry objects. Use only 1 eigenvalue for a conjugage pair
        self.eigenEntries = []
        if isEigenvecs:
            entries = su.eigenvects(self.jacobianMat)
            for entry in entries:
                eigenvalue= entry[0]
                if su.isComplex(eigenvalue):
                    _, imag = su.asRealImag(eigenvalue)
                    if imag < 0:
                        continue
                mul = entry[1]
                vectors = entry[2]
                self.eigenEntries.append(EigenEntry(eigenvalue, mul, vectors))
        else:
            # Ignore the eigenvectors
            for eigenvalue, mul in self.jacobianMat.eigenvals().items():
                self.eigenEntries.append(EigenEntry(eigenvalue, mul, None))

    def getJacobian(self, subs=None):
        """
        Retrieves the Jacobian matrix for the fixed, with optional substitutions.

        Parameters
        ----------
        subs: dict
        
        Returns
        -------
        sympy.Matrix
        """
        if subs is None:
            subs = {}
        return su.evaluate(self.jacobianMat, subs=subs, isNumpy=False)

    def getEigenvalues(self, subs=None):
        if subs is None:
            subs = {}
        eigenvalues = []
        [eigenvalues.extend(su.evaluate(e.getEigenvalues(), subs=subs))
              for e in self.eigenEntries]
        return eigenvalues


class ODEModel():

    def __init__(self, stateDct, initialDct=None, isEigenvecs=True):
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
        self.isEigenvecs = isEigenvecs
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
                return {v: p for p, v in zip(points, self.stateSymVec)}
            #
            if isinstance(fixedPoints, dict):
                fixedDcts = [mkDct(fixedPoints)]
            else:
                fixedDcts = []
                for fixedPoint in fixedPoints:
                    fixedDcts.append(mkDct(fixedPoint))
            return fixedDcts
        # Calculate the fixed points
        fixedPoints = sympy.solve(self.stateEprVec, self.stateSymVec)
        fixedDcts = mkFixedDcts(fixedPoints)
        return [FixedPoint(f, self.jacobianMat, isEigenvecs=self.isEigenvecs)
              for f in fixedDcts]

    def getFixedPointValues(self, subs=None):
        """
        Returns of values of all fixed points.

        Parameters
        ----------
        subs: dict
        
        Returns
        -------
        list-dict
            key: sympy.Symbol (state variable)
            value: float/expression
        """
        if subs is None:
            subs = {}
        return [{k: su.evaluate(v, subs=subs) for k, v in f.valueDct.items()}
              for f in self.fixedPoints]
