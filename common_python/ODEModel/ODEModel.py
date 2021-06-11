""" 
Model for non-linear ODE.

An ODEModel has one or more FixedPoint.
Each FixedPoint has a Jacobian and one or more EigenEntry.
An EigenEntry has 1 (if real) or 2 (if complex) eigenvalues
and one or more eigenvectors.
 """ 

import common_python.ODEModel.constants as cn
from common_python.sympy import sympyUtil as su

import copy
import numpy as np
from scipy import optimize
import sympy

X = "x"
t = sympy.Symbol("t")
IS_TIMER = False
LARGE_REAL = 10e6


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

    def equals(self, other):
        """
        Checks for same values in both EigenEntry.

        Parameters
        ----------
        other: EigenEntry
        
        Returns
        -------
        bool
        """
        if len(self.vectors) != len(other.vectors):
            return False
        isSame = self.isReal == other.isReal
        isSame = isSame and (self.value == other.value)
        isSame = isSame  \
              and (self.algebraicMultiplicity == other.algebraicMultiplicity)
        for vec, otherVec in zip(self.vectors, other.vectors):
            isSame = isSame and (vec.equals(otherVec))
        return isSame

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
        self.isEigenvecs = isEigenvecs
        # State at fixed point
        self.valueDct = {s: su.evaluate(valueDct[s], subs=subs) for s in valueDct.keys()}
        self.stateSymVec = sympy.Matrix(list(valueDct.keys()))
        self.jacobianMat = su.evaluate(jacobianMat, subs=valueDct, isNumpy=False)
        self.jacobianMat = su.evaluate(self.jacobianMat, subs=subs, isNumpy=False)
        # Create the EigenEntry objects. Use only 1 eigenvalue for a conjugage pair
        self.eigenEntries = []
        if self.isEigenvecs:
            entries = su.eigenvects(self.jacobianMat)
            for entry in entries:
                eigenvalue= entry[0]
                if su.isComplex(eigenvalue):
                    _, imag = su.asRealImag(eigenvalue)
                    if imag < 0:
                        continue
                mul = entry[1]
                vectors = entry[2]
                try:
                    eigenvalue = su.expressionToNumber(eigenvalue)
                except TypeError:
                    pass
                self.eigenEntries.append(EigenEntry(eigenvalue, mul, vectors))
        else:
            # Ignore the eigenvectors
            for eigenvalue, mul in self.jacobianMat.eigenvals().items():
                try:
                    newEigenvalue = su.expressionToNumber(eigenvalue)
                except TypeError:
                    newEigenvalue = eigenvalue
                    pass
                self.eigenEntries.append(EigenEntry(newEigenvalue, mul, None))

    def equals(self, other):
        """
        Determines if both fixed point have the same state.

        Parameters
        ----------
        other: FixedPoint
        
        Returns
        -------
        bool
        """
        if len(self.eigenEntries) != len(other.eigenEntries):
            return False
        #
        isSame = self.isEigenvecs == other.isEigenvecs
        trues = [self.valueDct[s] == other.valueDct[s]
              for s in self.stateSymVec]
        isSame = isSame and all(trues)
        isSame = isSame and self.stateSymVec.equals(other.stateSymVec)
        isSame = isSame and self.jacobianMat.equals(other.jacobianMat)
        for entry1, entry2 in zip(self.eigenEntries, other.eigenEntries):
            isSame = isSame and entry1.equals(entry2)
        return isSame

    def copy(self, subs=None):
        """
        Copies the fixed point, making optional substitutions.

        Parameters
        ----------
        subs: dict
        
        Returns
        -------
        FixedPoint
        """
        if subs is None:
            subs = {}
        valueDct = copy.deepcopy(self.valueDct)
        jacobianMat = copy.deepcopy(self.jacobianMat)
        newFixedPoint = self.__class__(valueDct, jacobianMat, subs=subs,
              isEigenvecs=self.isEigenvecs)
        return newFixedPoint
        

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
        [eigenvalues.extend(su.recursiveEvaluate(e.getEigenvalues(), subs=subs))
              for e in self.eigenEntries]
        # Filter possible redundancies because of subtitutions
        newEigenvalues = []
        for idx, eigenvalue1 in enumerate(eigenvalues[:-1]):
            isAdd = True
            for eigenvalue2 in eigenvalues[idx+1:]:
                try:
                    num = su.expressionToNumber(eigenvalue1 - eigenvalue2)
                    if np.isclose(num , 0):
                        isAdd = False
                        break
                except TypeError:
                    # Handle comparison of symbols
                    continue
            if isAdd:
                newEigenvalues.append(eigenvalue1)
        newEigenvalues.append(eigenvalues[-1])
        return newEigenvalues


class ODEModel():

    def __init__(self, stateDct, initialDct=None, isEigenvecs=True, isFixedPoints=True, subs=None):
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
        if subs is None:
            subs = {}
        self.isFixedPoints = isFixedPoints
        self.stateDct = {s: e.subs(subs) for s, e in stateDct.items()}
        if initialDct is not None:
            self.initialDct = {s: e.subs(subs) for s, e in initialDct.items()}
        self.isEigenvecs = isEigenvecs
        self.stateSymVec = sympy.Matrix(list(stateDct.keys()))
        self.stateEprVec = sympy.Matrix([self.stateDct[s]
              for s in self.stateSymVec])
        self.stateEprVec = su.evaluate(self.stateEprVec, subs=subs, isNumpy=False)
        self.jacobianMat = self.stateEprVec.jacobian(self.stateSymVec)
        self.fixedPoints = None
        if self.isFixedPoints:
            self.fixedPoints = self._calcFixedPoints(isEigenvecs=self.isEigenvecs)
        # Internal only
        self._bestFixedPoint = None
        self._bestParameterDct = None
        self._minReal = LARGE_REAL

    def _calcFixedPoints(self, subs={}, isEigenvecs=True):
        """
        Symbolically solves for fixed points for the model using the
        substitutions provided.

        Parameters
        ----------
        subs: dict
        isEigenvecs: bool
            Calculate eigenvectors

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
        stateEprVec = sympy.Matrix([e.subs(subs) for e in self.stateEprVec])
        jacobianMat = stateEprVec.jacobian(self.stateSymVec)
        fixedPoints = sympy.solve(stateEprVec, self.stateSymVec)
        fixedDcts = mkFixedDcts(fixedPoints)
        return [FixedPoint(f, jacobianMat, isEigenvecs=isEigenvecs)
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

    def findOscillations(self, parameterSyms, minImag=1.0, **kwargs):
        """
        Finds values of model parameters that result in oscillations
        using the optimization "Minimize the real part of the Eigenvalue
        subject to a constraint on the imaginary part."
        Methods are:
          'nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc',
           'cobyla', 'slsqp'

        Parameters
        ----------
        parameters: list-str
        minImag: minimum value of the imaginary part of the eigenvalue
        kwargs: dict
            optional arguments to scipy.optimize
            default: method="nelder-mead"
        
        Returns
        -------
        FixedPoint
            FixedPoint that has an eigenvalue meeting the criteria
        dict
            Dictionary of parameter values used to calculate FixedPoint
        """
        if len(kwargs) == 0:
            kwargs["method"] = "nelder-mead"
        method = kwargs["method"]
        self._minReal = LARGE_REAL
        self._bestFixedPoint = None
        self._bestParameterDct = None
        #
        def mkValueDct(values):
            return {s: v for s, v in zip(parameterSyms, values)}
        #
        def calcLoss(values):
            """
            Calculates eigenvalues for the parameter values provided.
            Returns the squared real part of the eigenvalue with the largest
            imaginary part.

            Parameters
            ----------
            values: list-float
                values that correspond to the parameters parameterSyms
            
            Returns
            -------
            float
            """
            dct = mkValueDct(values)
            minReal = LARGE_REAL
            # Find the best possibility of an oscillating eigenvector
            for fixedPoint in self.fixedPoints:
                newFixedPoint = fixedPoint.copy(subs=dct)
                for entry in newFixedPoint.eigenEntries:
                    real, imag = su.asRealImag(entry.value)
                    absReal = np.abs(real)
                    if imag < minImag:
                        continue
                    if absReal < minReal:
                        minReal = absReal
            # Update best found if needed
            if minReal < self._minReal:
                self._minReal = minReal
                self._bestParameterDct = mkValueDct(values)
                self._bestFixedPoint = newFixedPoint
            return minReal
        #
        initialValues = np.repeat(1, len(parameterSyms))
        if method in ('nelder-mead', 'powell', 'anneal', 'cobyla'):
            jac = None
        else:
            jac = calcLoss
        solution = optimize.minimize(calcLoss, initialValues, jac=jac,
              tol=1e-5, **kwargs)
        valueDct = mkValueDct(solution.x)
        return self._bestFixedPoint, self._bestParameterDct

    @classmethod
    def mkODEModel(cls, roadRunner):
        """
        Creates an ODEModel from a roadrunner instance.

        Parameters
        ----------
        roadRunner: ExtendedRoadrunner
        
        Returns
        -------
        ODEModel
        """

