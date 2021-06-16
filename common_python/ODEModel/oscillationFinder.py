""" 
Finds values of parameters in a roadrunner model that have oscillations.
 """ 

import common_python.ODEModel.constants as cn
from src.common.simple_sbml import SimpleSBML

import collections
import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


LARGE_REAL = 1e6


class FeasibilityFoundException(Exception):
    pass


class XDict(dict):

    """Values that are accessed by a keyword."""

    def __init__(self, names=None, values=None):
        """
        Parameters
        ----------
        names: list-str / None
        values: list-float / None/ float
        """
        self._make(names, values)

    def _make(self, names, values):
        """
        Sets values provided in constructor.

        Parameters
        ----------
        names: list-str / None
        values: list-float / None/ float
        """
        def isList(o):
            try:
                lst = [x for x in o]
                return True
            except Exception:
                return False
        #
        if names is None:
            return None
        if not isList(names):
            raise ValueError("names must be a list of strings or None")
        if isList(values):
            if len(names) != len(values):
                msg = "names and values must have same length if values!=float."
                raise ValueError(msg)
            dct = {n: v for n, v in zip(names, values)}
        else:
            dct = {n: values for n in names}
        for key, value in dct.items():
            self.__setitem__(key, value)

    @classmethod
    def mkParameter(cls, roadrunner):
         """
         Creates a XDict for parameters in roadrunner.
 
         Parameters
         ----------
         roadrunner: ExtendedRoadrunner
         
         Returns
         -------
         XDict
         """
         simple = SimpleSBML(roadrunner)
         names = [p.id for p in simple.parameters]
         values = [roadrunner[n] for n in names]
         return cls(names=names, values=values)
 
    @classmethod
    def mkSpecies(cls, roadrunner):
         """
         Creates a XDict for species in roadrunner.
 
         Parameters
         ----------
         roadrunner: ExtendedRoadrunner
         
         Returns
         -------
         XDict
         """
         simple = SimpleSBML(roadrunner)
         names = [s.id for s in simple.species]
         values = [roadrunner[n] for n in names]
         return cls(names=names, values=values)

    def equals(self, other):
        """
        Tests if floating point values are equal.

        Parameters
        ----------
        other: XDict
        
        Returns
        -------
        bool
        """
        diffKey = set(self.keys()).symmetric_difference(other.keys())
        if len(diffKey) > 0:
            return False
        # Has the same keys
        trues = [np.isclose(self[k], other[k]) for k in self.keys()]
        return all(trues)


class OscillationFinder():

    def __init__(self, roadrunner):
        """
        Parameters
        ----------
        roadrunner: ExtendedRoadrunner
        """
        self.roadrunner = roadrunner
        self.parameterXD = XDict.mkParameter(self.roadrunner)
        self.parameterNames = list(self.parameterXD.keys())
        self.simulationArr = None  # Results from last setSteadyState
        # Internal only
        self._bestFixedPoint = None
        self._bestParameterXD = None
        self._minReal = LARGE_REAL

    def setSteadyState(self, parameterXD=None):
        """
        Puts the simulation into steady state for the parameter values.

        Parameters
        ----------
        parameterXD: dict
            key: Name
            value: value
        
        Returns
        -------
        XDict: fixed point
        """
        self.simulate(parameterXD=parameterXD)
        self.roadrunner.getSteadyStateValues()
        return XDict.mkSpecies(self.roadrunner)

    def setParameters(self, parameterXD=None):
        """
        Sets the parameter values in the simulation.

        Parameters
        ----------
        parameterXD: dict
            key: Name
            value: value
        """
        if parameterXD is None:
            parameterXD = self.parameterXD
        for name, value in parameterXD.items():
            self.roadrunner[name] = value

    def simulate(self, parameterXD=None):
        """
        Runs a simulation for a set of parameter values.

        Parameters
        ----------
        parameterXD: dict
            key: Name
            value: value
        """
        self.roadrunner.reset()
        self.setParameters(parameterXD=parameterXD)
        self.simulationArr = self.roadrunner.simulate()

    def _getEigenvalues(self, parameterXD=None):
        _ = self.setSteadyState(parameterXD=parameterXD)
        return np.linalg.eig(self.roadrunner.getFullJacobian())[0]

    def find(self, minImag=1.0, minReal=0.1, **kwargs):
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
        minReal: minimum value of real for feasibility
        kwargs: dict
            optional arguments to scipy.optimize
            default: method="nelder-mead"
        
        Returns
        -------
        parameterXD / None (none found)
            key: parameter name
            value: value of the parameter
        """
        # TODO: Stop if encounter a feasible solution
        #       Generalize to feasibility search?
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
            parameterXD = XDict(names=self.parameterNames, values=values)
            eigenvalues = self._getEigenvalues(parameterXD=parameterXD)
            # See how close to meeting criteria
            isDone = False
            candidateImag = -LARGE_REAL
            candidateReal = -LARGE_REAL
            for entry in eigenvalues:
                real, imag = np.real(entry), np.imag(entry)
                if imag < minImag:
                    continue
                candidateImag = imag
                candidateReal = real
                if candidateReal < minReal:
                    continue
                isDone = True
                break
            # Update best found if needed
            if isDone:
                self._bestParameterXD = XDict(self.parameterNames, values)
                raise FeasibilityFoundException
            if candidateImag < minImag:
                # Not feasible in terms of imaginary part
                return LARGE_REAL
            else:
                # Feasible imaginary part, but real is too small
                return minReal - candidateReal
        #
        if len(kwargs) == 0:
            kwargs["method"] = "nelder-mead"
        method = kwargs["method"]
        self._minReal = LARGE_REAL
        self._bestParameterXD = None
        #
        if method in ('nelder-mead', 'powell', 'anneal', 'cobyla'):
            jac = None
        else:
            jac = _calcLoss
        try:
            _ = optimize.minimize(_calcLoss, list(self.parameterXD.values()),
                  jac=jac,
                  tol=1e-5, **kwargs)
        except FeasibilityFoundException:
            pass
        return self._bestParameterXD

    def plot(self, isPlot=True):
        """
        Plots the results of the last simulation.

        Parameters
        ----------
        isPlot: bool
        """
        _, ax = plt.subplots(1)
        numSpecies = len(self.simulationArr.colnames) - 1
        for idx in range(numSpecies):
            ax.plot(self.simulationArr[:, 0], self.simulationArr[:, idx+1])
        ax.legend(self.simulationArr.colnames[1:])
        ax.set_xlabel("time")
        if isPlot:
            plt.show()
