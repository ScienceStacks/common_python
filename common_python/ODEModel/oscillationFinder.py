""" 
Finds values of parameters in a roadrunner model that have oscillations.
 """ 

import common_python.ODEModel.constants as cn
from src.common.simple_sbml import SimpleSBML

import collections
import copy
import numpy as np
from scipy import optimize


class OscillationFinder():

    def __init__(self, roadrunner):
        """
        Parameters
        ----------
        roadrunner: ExtendedRoadrunner
        """
        self.roadrunner = roadrunner
        self.parameterDct = self._mkParameterDct()  # Original parameter values
        self.parameterNames = list(self.parameterDct.keys())
        # Internal only
        self._bestFixedPoint = None
        self._bestParameterDct = None
        self._minReal = LARGE_REAL

    def setSteadyState(self, parameterDct=None):
        """
        Puts the simulation into steady state for the parameter values.

        Parameters
        ----------
        parameterDct: dict
            key: Name
            value: value
        
        Returns
        -------
        dict: fixedPointDct
        """
        self.roadrunner.reset()
        if parameterDct is None:
            parameterDct = self.parameterDct
        # Set the parameter values
        for name, value in parameterDct.items():
            self.roadrunner[name] = value
        self.roadrunner.getSteadyState()
        # Retrieve the values of floating species
        for species in self.simple.species:
            dct[species.id] = self.roadrunner[species.id]
        return dct

    def _mkParameterDct(self, values=None):
        """
        Creates a dictionary of parameter values for the simulation.

        Parameters
        ----------
        values: list-float
        
        Returns
        -------
        dict
            key: name
            value: float
        """
        if values is None:
            simple = SimpleSBML(roadrunner)
            parameterDct = {p.id: p.value for p in simple.parameters}
        else:
            parameterDct = {n: v for n, v in zip(self.parameterNames, values)}
        return parameterDct

    def _getEigenvalues(self, parameterDct):
        _ = self.setSteadyState(parameterDct=parameterDct)
        return self.roadrunner.getEigenvalues()

    @staticmethod
    def _calcLoss(values):
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
        parameterDct = self._mkParameterDct(values=value)
        eigenvalues = self._getEigenvalues(parameterDct)
        # See how close to meeting criteria
        for entry in eigenvalues:
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

    def find(self, minImag=1.0, **kwargs):
        """
        Finds values of model parameters that result in oscillations
        using the optimization "Minimize the real part of the Eigenvalue
        subject to a constraint on the imaginary part."
        Methods are:
          'nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc',
           'cobyla', 'slsqp'

        Parameters
        ----------
        minImag: minimum value of the imaginary part of the eigenvalue
        kwargs: dict
            optional arguments to scipy.optimize
            default: method="nelder-mead"
        
        Returns
        -------
        parameterDct / None (none found)
            key: parameter name
            value: value of the parameter
        """
        if len(kwargs) == 0:
            kwargs["method"] = "nelder-mead"
        method = kwargs["method"]
        self._minReal = LARGE_REAL
        self._bestFixedPoint = None
        self._bestParameterDct = None
        #
        def mkValueDct(values):
            return {s: v for s, v in zip(self.parameterDct, values)}
        #
        #
        if method in ('nelder-mead', 'powell', 'anneal', 'cobyla'):
            jac = None
        else:
            jac = calcLoss
        solution = optimize.minimize(calcLoss, self.parameterDct.values(), jac=jac,
              tol=1e-5, **kwargs)
        valueDct = mkValueDct(solution.x)
        return self._bestFixedPoint, self._bestParameterDct
