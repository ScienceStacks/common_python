""" 
Finds values of parameters in a roadrunner model that have oscillations.
 """ 

import common_python.ODEModel.constants as cn
from src.common.simple_sbml import SimpleSBML

import collections
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import optimize
import seaborn as sns
import tellurium as te


LARGE_REAL = 1e6
MAX_ENDTIME = 1000


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
    def mkParameters(cls, roadrunner):
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
        self.parameterXD = XDict.mkParameters(self.roadrunner)
        self.parameterNames = list(self.parameterXD.keys())
        self.simulationArr = None  # Results from last setSteadyState
        # Internal only
        self._bestParameterXD = None
        self.bestEigenvalue = None
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
        if self.simulationArr is None:
            return None
        try:
            self.roadrunner.getSteadyStateValues()
        except RuntimeError:
            return None
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

    def simulate(self, parameterXD=None, endTime=10):
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
        try:
            self.simulationArr = self.roadrunner.simulate(0, endTime, 5*endTime)
        except RuntimeError:
            self.simulationArr = None
            

    def _getEigenvalues(self, parameterXD=None):
        _ = self.setSteadyState(parameterXD=parameterXD)
        if self.simulationArr is None:
            return None
        return np.linalg.eig(self.roadrunner.getFullJacobian())[0]

    def find(self, initialParameterXD=None, lowerBound=0, upperBound=1e3,
          minImag=0.1, minReal=0.0, maxReal=0.1, maxDeficiency=0.2, **kwargs):
        """
        Finds values of model parameters that result in oscillations
        using the optimization "Minimize the real part of the Eigenvalue
        subject to a constraint on the imaginary part."
        Methods are:
          'nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc',
           'cobyla', 'slsqp'

        Parameters
        ----------
        initialParameterXD: XDict
        lowerBound: float
              lower bound for parameter search
        upperBound: float
              upper bound for parameter search
        minImag: float
             minimum value of the imaginary part of the eigenvalue
             hard constraint
        minReal: Float
             minimum value of real for feasibility
             soft constraint
        maxReal: Float
             maximum value of real for feasibility
             soft constraint
        maxDeficiency: Float
             maximum deficiency to be acceptable
        kwargs: dict
            optional arguments to scipy.optimize
            default: method="nelder-mead"
        
        Returns
        -------
        parameterXD / None (none found)
            key: parameter name
            value: value of the parameter
        """
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
            def calcDeficiency(eigenvalue):
                """
                Scores the eigenvalue in terms of how far it is from
                a feasible value.               

                Parameters
                ----------
                eigenvalue: complex
                
                Returns
                -------
                float
                """
                real, imag = np.real(eigenvalue), np.imag(eigenvalue)
                if (imag < minImag):
                    return LARGE_REAL
                deficiency = max(minReal - real, real - maxReal)
                return max(0, deficiency)
            #
            parameterXD = XDict(names=self.parameterNames, values=values)
            eigenvalues = self._getEigenvalues(parameterXD=parameterXD)
            if eigenvalues is None:
                return LARGE_REAL
            # See how close to meeting criteria
            self.bestEigenvalue = None
            bestDeficiency = LARGE_REAL
            for eigenvalue in eigenvalues:
                deficiency = calcDeficiency(eigenvalue)
                if deficiency < bestDeficiency:
                    # Verify that these are simulateable parameters
                    self.simulate(parameterXD=parameterXD, endTime=MAX_ENDTIME)
                    if self.simulationArr is not None:
                        self.bestEigenvalue = eigenvalue
                        bestDeficiency = deficiency
                if deficiency == 0:
                    break
            # Update best found if needed
            if bestDeficiency <= maxDeficiency:
                self._bestParameterXD = XDict(self.parameterNames, values)
                raise FeasibilityFoundException
            return bestDeficiency
        #
        self._minReal = LARGE_REAL
        self._bestParameterXD = None
        if initialParameterXD is None:
            initialParameterXD = self.parameterXD
        bounds = [(lowerBound, upperBound) for _ in range(len(initialParameterXD))]
        #
        try:
            _ = optimize.differential_evolution(_calcLoss, bounds, **kwargs)
        except FeasibilityFoundException:
            pass
        # Determine if search was successful
        return self._bestParameterXD

    def plotTime(self, title="", ylim=None, isPlot=True,
          startTime=0, endTime=None):
        """
        Plots the results of the last simulation.

        Parameters
        ----------
        isPlot: bool
        """
        def findTimeIdx(value):
            arr = (self.simulationArr[:, 0] - value)**2
            minDiff = min(arr)
            return arr.tolist().index(minDiff)
        #
        if endTime is None:
            endTime = self.simulationArr[0, -1]
        startIdx = findTimeIdx(startTime)
        endIdx = findTimeIdx(endTime)
        #
        _, ax = plt.subplots(1)
        numSpecies = len(self.simulationArr.colnames) - 1
        for idx in range(numSpecies):
            ax.plot(self.simulationArr[startIdx:endIdx, 0],
                  self.simulationArr[startIdx:endIdx, idx+1])
        ax.legend(self.simulationArr.colnames[1:])
        ax.set_xlabel("time")
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(ylim)
        if isPlot:
            plt.show()
    
    @classmethod
    def analyzeFile(cls, modelPath, outPath="analyzeFile.csv",
          numRestart=2, isPlot=True, **kwargs):
        """
        Finds parameters with oscillations for a single file.
        Plots simulations for the optimized parameters.
        
        Parameters
        ----------
        modelPath: str
        numRestart: int
            Number of times a search is redone if unsuccessful
        isPlot: bool
        outPath: str
            Saves results with file structured as:
                modelID: str
                originalParameterDct: dict as str
                newParameterDct: dict as str
                eigenvalue: complex
                foundOscillations: bool
        kwargs: dict
            parameters passed to plot
        
        Returns
        -------
        XDict
            feasible parameters found
        bool
            verified oscillations
        """
        # Construct the model ID
        start = modelPath.index("Model_") + len("Model_")
        end = modelPath.index(".ant")
        modelId = modelPath[start:end]
        #
        def mkTitle(string):
            return "%s: %s" % (modelId, string)
        #
        def writeResults(originalParameterXD, feasibleParameterXD,
              eigenvalue, foundOscillations):
            real, imag = eigenvalue.real, eigenvalue.imag
            line = "\n%s, '%s', '%s', %3.2f + %3.2fj %s" % (
                  modelId, str(originalParameterXD), 
                        str(feasibleParameterXD),
                        real, imag,
                        str(foundOscillations))
            with open(outPath, "a") as fd:
                fd.writelines(line)
        #
        def plot(finder, parameterXD, title):
            if not isPlot:
                return
            isPlotted = False
            if parameterXD is not None:
                finder.simulate(parameterXD=parameterXD, endTime=100)
                if finder.simulationArr is not None:
                    finder.plotTime(title=mkTitle(title), isPlot=isPlot)
                    isPlotted = True
            if not isPlotted:
                msg = "No simulation produced for parameters of %s!" % title
                print(mkTitle(msg))
            return isPlotted
        # 
        for idx in range(numRestart + 1):
            rr = te.loada(modelPath)
            finder = OscillationFinder(rr)
            if idx == 0:
                # Plot simulation of the original parameters
                finder.simulate()
                originalParameterXD = XDict.mkParameters(finder.roadrunner)
                plot(finder, originalParameterXD, "Original Algorithm")
                if False:
                    # Change the parameter values
                    initialParameterXD = XDict.mkParameters(rr)
                    initialParameterXD = XDict(initialParameterXD.keys(), 1)
                    # finder.setParameters(initialParameterXD)
                    plot(finder, initialParameterXD, "Initial Parameters")
            # Find the parameters
            feasibleParameterXD = finder.find(minReal=0.0)
            if feasibleParameterXD is not None:
                break
        foundOscillations = plot(finder, feasibleParameterXD,
              "Optimized Parameters")
        writeResults(originalParameterXD, feasibleParameterXD,
              finder.bestEigenvalue,
              foundOscillations is not None)
        #
        return feasibleParameterXD, foundOscillations

    def plotJacobian(self, fig=None, ax=None, isPlot=True, isLabel=True, title="",
        **kwargs):
        """
        Constructs a hetmap for the jacobian. The Jacobian must be purely
        sumeric.

        Parameters
        ----------
        isPlot: bool
        ax: Matplotlib.axes
        isLabel: bool
            include labels
        kwargs: dict
            optional keywords for heatmap
        """
        try:
            self.roadrunner.getSteadyStateValues()
            mat = self.roadrunner.getFullJacobian()
        except RuntimeError:
            return
        colnames = mat.colnames
        rownames = list(colnames)
        df = pd.DataFrame(mat)
        if isLabel:
            df.columns = colnames
            df.index = rownames
        else:
            df.columns = np.repeat("", len(df.columns))
            df.index = np.repeat("", len(df.columns))
        df = df.applymap(lambda v: float(v))
        # Scale the entries in the matrix
        maxval = df.max().max()
        minval = df.min().min()
        maxabs = max(np.abs(maxval), np.abs(minval))
        # Plot
        if ax is None:
            fig, ax = plt.subplots(1)
        sns.heatmap(df, cmap='seismic', ax=ax, vmin=-maxabs, vmax=maxabs,
              **kwargs)
        #ax.text(0.9, 0.5, title)
        ax.set_title(title)
        if isPlot:
            plt.show()

    @classmethod
    def plotJacobians(cls, antimonyPaths, isPlot=True, numRowCol=None,
          figsize=(12, 10), **kwargs):
        """
        Constructs heatmaps for the Jacobians for a list of files.

        Parameters
        ----------
        antimonyPaths: list-str
            Each path name has the form *Model_<id>.ant.
        numRowCol: (int, int)
        kwargs: dict
            Passed to plotter for each file
        """
        if "isPlot" in kwargs:
            del kwargs["isPlot"]
        def mkTitle(path, string):
            start = path.index("Model_") + len("Model_")
            end = path.index(".ant")
            prefix = path[start:end]
            if len(string) == 0:
                title = "%s" % prefix
            else:
                title = "%s: %s" % (prefix, string)
            return title
        #
        numPlot = len(antimonyPaths)
        if numRowCol is None:
            numRow = np.sqrt(numPlot)
            if int(numRow) < numRow:
                numRow = int(numRow) + 1
            else:
                numRow = int(numRow)
            numCol = numRow
        else:
            numRow = numRowCol[0]
            numCol = numRowCol[1]
        fig, axes = plt.subplots(numRow, numCol, figsize=figsize)
        countPlot = 1
        for irow in range(numRow):
            for icol in range(numCol):
                if countPlot > numPlot:
                    break
                # Colorbar
                if (irow == 0) and (icol == numCol - 1):
                    cbar = True
                else:
                    cbar = False
                # Labels
                if (irow == numRow - 1)  and (icol == 0):
                    isLabel = True
                else:
                    isLabel = False
                # Construct plot
                antimonyPath = antimonyPaths[countPlot-1]
                roadrunner = te.loada(str(antimonyPath))
                #title = mkTitle(antimonyPath, "")
                title = str(countPlot - 1)
                # Construct the ODEModel and do the plot
                finder = cls(roadrunner)
                finder.plotJacobian(isPlot=False, ax=axes[irow, icol],
                      isLabel=isLabel, title=title, cbar=cbar,  fig=fig,
                      cbar_kws=dict(use_gridspec=False,location="top"),
                      **kwargs)
                countPlot += 1
        if isPlot:
            plt.show()
