""" 
Model for non-linear ODE.

An ODEModel has one or more FixedPoint.
Each FixedPoint has a Jacobian and one or more EigenEntry.
An EigenEntry has 1 (if real) or 2 (if complex) eigenvalues
and one or more eigenvectors.
 """ 

import common_python.ODEModel.constants as cn
from common_python.sympy import sympyUtil as su

import collections
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
import seaborn as sns
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

    def __init__(self, stateDct, initialDct=None,
          isEigenvecs=True, isFixedPoints=True, subs=None):
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

    @classmethod
    def mkODEModel(cls, roadrunner, **kwargs):
        from src.common.simple_sbml import SimpleSBML
        """
        Creates an ODEModel from a roadrunner instance.

        Parameters
        ----------
        roadrunner: ExtendedRoadrunner
        kwargs: dict
              optional arguments for the constructor
        
        Returns
        -------
        ModelInfo
        """
        # mdl: ODEModel
        # stm: stoichiometry matrix
        # sre: species reaction expression dictionary
        #      (species flux expressed in terms of reaction fluxes)
        ModelInfo = collections.namedtuple("ModelInfo", "mdl stm sre")
        #
        simple = SimpleSBML(roadrunner)
        # Add the required names
        names = [p.id for p in simple.parameters]
        speciesNames = [s.id for s in simple.species]
        names.extend(speciesNames)
        for reaction in simple.reactions:
            name = reaction.id
            if name[0] == "_":
                name = name[1:]
            reaction.id = name
        reactionNames = [r.id for r in simple.reactions]
        names.extend(reactionNames)
        nameStr = " ".join(names)
        su.addSymbols(nameStr, dct=globals())
        # Create sympy expressions for kinetic laws
        reactionEprDct = {eval(r.id): eval(r.kinetic_law.formula)
              for r in simple.reactions}
        # Express species fluxes in terms of reaction fluxes
        reactionVec = sympy.Matrix(list(reactionEprDct.keys()))
        stoichiometryMat = sympy.Matrix(roadrunner.getFullStoichiometryMatrix())
        speciesReactionVec = stoichiometryMat * reactionVec
        speciesReactionDct = {eval(s): e for s, e in 
              zip(speciesNames, speciesReactionVec)}
        # FIXME: Need a better solver for fixed points, either
        #        doing substitution within state equations or
        #        using the reaction fluxes
        systemDct = {s: sympy.simplify(speciesReactionDct[s].subs(reactionEprDct))
            for s in speciesReactionDct.keys()}
        model = cls(systemDct, **kwargs)
        modelInfo = ModelInfo(mdl=model, stm=stoichiometryMat, sre=speciesReactionDct)
        return modelInfo

    @classmethod
    def findFixedPoints(cls, rr):
        """
        Finds the fixed points of the roadrunner model that only has mass
        action kinetics and no more than two reactants. 
        The approach uses relaxation by transforming the quadratic
        equations to a higher dimension linear space.

        This may run indefinitely depending on the structure of the equations.
        
        Parameteters
        ------------
        rr: ExtendedRoadRunner
        
        Returns
        -------
        list-dict
            key: state variable symbol
            value: expressions
        """
        def findIdx(sym):
            return list(relaxationResult.vec).index(sym)
        #
        modelInfo = ODEModel.mkODEModel(rr, isFixedPoints=False, isEigenvecs=False)
        relaxationResult = su.mkQuadraticRelaxation(modelInfo.mdl.stateDct)
        mat = sympy.simplify(relaxationResult.mat)
        nullVecs = mat.nullspace(relaxationResult.vec)
        # Create solution for the nullspace
        constantStrs = ["c%d" % n for n in range(len(nullVecs))]
        su.addSymbols(" ".join(constantStrs), dct=globals())
        constantSyms = [globals()[s] for s in constantStrs]
        mat =  sympy.Matrix(nullVecs)
        numRow = nullVecs[0].rows
        numCol = len(nullVecs)
        nullspaceLst = [list(v) for v in nullVecs]
        mat = sympy.Matrix(nullspaceLst)
        mat = mat.transpose()
        constantVec = sympy.Matrix(constantSyms)
        nullspaceVec = mat * constantVec
        # Solve for the constraints
        symEntryDct = {s: nullspaceVec[n] for n, s in enumerate(relaxationResult.vec)}
        # Construct equality expressions and solve.
        eprs = []
        for quadSym, quadEpr in relaxationResult.sub.items():
            quadIdx = findIdx(quadSym)
            syms = list(quadEpr.args)
            if syms[1] == 2:
                syms[1] = syms[0]
            symIdxs = [findIdx(s) for s in syms]
            prodEpr = nullspaceVec[symIdxs[0]] * nullspaceVec[symIdxs[1]]
            eprs.append(nullspaceVec[quadIdx] - prodEpr)
        # Values of constants to match constraints
        solutions = sympy.solve(eprs, constantSyms)
        # Calculate the fixed points
        simpleSyms = set(relaxationResult.vec).symmetric_difference(
               relaxationResult.sub.keys())
        simpleSyms = sorted(simpleSyms, key=lambda s: s.name)
        fixedPointDcts = []
        for solution in solutions:
            # Calculate the vector for each solution
            numVec = len(nullspaceLst)
            vec = sympy.zeros(len(relaxationResult.vec), 1)
            for idx in range(numVec):
                vec += solution[idx] * sympy.Matrix(nullspaceLst[idx])
            # Create the fixed point from the vector
            fixedPointDcts.append({simpleSyms[n]: vec[n]
                  for n in range(len(simpleSyms))})
        return fixedPointDcts

    def plotJacobian(self, isPlot=True, ax=None, isLabel=True, title=""):
        """
        Constructs a hetmap for the jacobian. The Jacobian must be purely
        sumeric.

        Parameters
        ----------
        isPlot: bool
        ax: Matplotlib.axes
        subs: dict
        isLabel: bool
            include labels
        """
        mat = self.jacobianMat.subs(subs)
        newMat = mat.subs(subs)
        if len(newMat.free_symbols) != 0:
            raise ValueError("Jacobian cannot have symbols.")
        df = pd.DataFrame(newMat.tolist())
        if isLabel:
            df.columns = [s.name for s in self.stateSymVec]
            df.index = df.columns
        df = df.applymap(lambda v: float(v))
        # Scale the entries in the matrix
        maxval = df.max().max()
        minval = df.min().min()
        maxabs = max(np.abs(maxval), np.abs(minval))
        # Plot
        if ax is None:
            _, ax = plt.subplots(1)
        sns.heatmap(df, cmap='seismic', ax=ax, vmin=-maxabs, vmax=maxabs)
        ax.set_title(title)
        if isPlot:
            plt.show()

    @classmethod
    def plotJacobians(cls, antimonyPaths, **kwargs):
        """
        Constructs heatmaps for the Jacobians for a list of files.

        Parameters
        ----------
        antimonyPaths: list-str
            Each path name has the form *Model_<id>.ant.
        kwargs: dict
            Passed to plotter for each file
        """
        def mkTitle(string):
            start = path.index("Model_") + len("Model_")
            end = path.index(".ant")
            prefix = path[start:end]
            if len(string) == 0:
                return "%s" % prefix
            else:
                return "%s: %s" % (prefix, string)
            return "%s: %s" % (prefix, string)
        #
        numPlot = len(antimonyPaths)
        numRow = np.sqrt(numPlot)
        if int(numRow) < numRow:
            numRow = int(numRow) + 1
        else:
            numRow = int(numRow)
        fig, axes = plt.subplots(numRow, numRow)
        countPlot = 1
        for irow in range(numRow):
            for icol in range(numcol):
                if countPlot > numPlot:
                    break
                antimonyPath = antimonyPaths[countPlot]
                roadrunner = te.loada(antimonyPath)
                title = mkTitle("")
                # Create the substitutions for floating species, fixed speceis, and parameters
                # TODO:
                # Construct the ODEModel and do the plot
                odeModel = ODEModel.mkODEModel(roadrunner,
                      isEigenvecs=False, isFixedPoints=False)
                odeModel.plotJacobian(isPlot=True, ax=axes[irow, icol], subs=subs,
                      isLabel=False, title=mkTitle(antimonyPath), **kwargs)
                countPlot += 1
                
