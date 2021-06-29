""" Model of linear time-invariant systems.

SUffix conventions
  *Arr - numpy array
  *Mat - sympy.Matrix N X N
  *Vec - sympy.Matrix N X 1
  *s - list

"""
from common_python.util.timer import Timer
from common_python.sympy import sympyUtil as su
from common_python.ODEModel.eigenCollection import EigenCollection

import matplotlib.pyplot as plt
import sympy

X = "x"
t = sympy.Symbol("t")
IS_TIMER = False


class LTIModel():

    def __init__(self, aMat, initialVec, rVec=None):
        """
        Parameters
        ----------
        aMat: sympy.Matrix (N X N)
            A marix
        initialVec: sympy.Matrix (N X 1)
            Initial values
        rVec: sympy.Matrix (N X 1)
            r matrix in the differential equation
        """
        self.aMat = aMat
        self.numRow = self.aMat.rows
        self.initialVec = initialVec
        self.rVec = rVec
        # Results
        self.eigenCollection = EigenCollection(self.aMat)
        self.eigenCollection.completeEigenvectors()
        self.fundamentalMat = None
        self.coefHomogeneousVec = None
        self.particularVec = None
        self.solutionVec = None
        self.evaluatedSolutionVec = None

    def solve(self, subs=None):
        """
        Solves the LTI system symbolically.
        Updates self.solutionVec.

        Returns
        -------
        sympy.Matrix (N X 1)
        """
        if subs is None:
            subs = {}
        # TODO: Handle imaginary eigenvalues
        timer = Timer("solve", isEnable=IS_TIMER)
        # Do evaluations
        aMat = su.evaluate(self.aMat, isNumpy=False, subs=subs)
        self.eigenCollection = EigenCollection(aMat)
        initialVec = su.evaluate(self.initialVec, isNumpy=False, subs=subs)
        # Find the complete set of eigenvectors, handling cases
        # in which the algebraic multiplicity > geometric multiplicity
        timer.print(name="solve_1")
        # Construct vector of solutions
        vecs = []
        eigenInfos = self.eigenCollection.pruneConjugates()
        for eigenInfo in eigenInfos:
            for eigenVector in eigenInfo.vecs:
                if su.isReal(eigenInfo.val) or su.isSympy(eigenInfo.val):
                    expr = sympy.exp(eigenInfo.val * t)
                    vecs.append(eigenVector * expr)
                else:
                    if eigenInfo.mul > 1:
                        msg = "Large multiciplies for complex eigenvalues."
                        raise RuntimeError(msg)
                    # Add two vectors since the conjugate eigenvalues are
                    # done together
                    realEig, imagEig = su.asRealImag(eigenInfo.val)
                    realVec, imagVec = su.vectorAsRealImag(eigenVector)
                    expEx = sympy.exp(realEig * t)
                    cosEx = sympy.cos(imagEig * t)
                    sinEx = sympy.sin(imagEig * t)
                    expr =  expEx * (realVec * cosEx - imagVec * sinEx)
                    vecs.append(expr)
                    expr =  expEx * (realVec * sinEx + imagVec * cosEx)
                    vecs.append(expr)
        # Construct the fundamental matrix
        self.fundamentalMat= sympy.Matrix(vecs)
        self.fundamentalMat = self.fundamentalMat.reshape(self.numRow, self.numRow)
        self.fundamentalMat = sympy.simplify(self.fundamentalMat.transpose())
        timer.print(name="solve_2")
        # Find the coefficients for the homogeneous system
        # This is done without computing an inverse since the A matrix
        # may be singular.
        t0Mat = sympy.simplify(self.fundamentalMat.subs(t, 0)) # evaluate at time 0
        self.coefHomogeneousVec = su.solveLinearSingular(t0Mat, initialVec)
        timer.print(name="solve_3")
        # Particular solution
        if self.rVec is not None:
            rVec = su.evaluate(self.rVec, isNumpy=False, subs=subs)
            invfundMat = self.fundamentalMat.inv()
            timer.print(name="solve_4a")
            dCoefMat = invfundMat*rVec
            timer.print(name="solve_4b")
            coefMat = sympy.integrate(dCoefMat, t)
            timer.print(name="solve_4c")
            self.particularVec = self.fundamentalMat * coefMat
            timer.print(name="solve_4")
        else:
            self.particularVec = sympy.zeros(self.numRow, 1)
            timer.print(name="solve_5")
        # Full solution
        self.solutionVec =  self.particularVec \
              + self.fundamentalMat*self.coefHomogeneousVec
        timer.print(name="solve_6")
        self.solutionVec = sympy.simplify(self.solutionVec)
        timer.print(name="solve_7")
        return self.solutionVec

    def evaluate(self, subs=None, **kwargs):
        """
        Returns a numerical solution.

        Parameters
        ----------
        kwargs: dict
            dictionary of symbol value assignments

        Returns
        -------
        numpy.ndarray (N X 1)
        """
        if subs is None:
            subs = {}
        if self.solutionVec is None:
            _ = self.solve(subs=subs)
        return su.evaluate(self.solutionVec, subs=subs, **kwargs)

    def plot(self, startTime, endTime, numPoint, subs, isPlot=True,
          ylabel="", title=""):
        """
        Does time series plot. Should not substitute 't'.

        Parameters
        ----------
        startTime: float
        endTime: float
        numPoint: int
        subs: dict
            arguments for evaluate
        isPlot: bool
            display the plot
        ylabel: str
        title: str
        """
        # Create the plot data
        vec = self.evaluate(subs=subs, isNumpy=False)
        delta = (endTime - startTime)/numPoint
        tVals = [startTime + n*delta for n in range(numPoint)]
        arrs = [vec.subs(t, tval) for tval in tVals]
        # Plot the results
        labels = ["x_%d" % n for n in range(len(vec))]
        _, ax = plt.subplots(1)
        legends = []
        for idx in range(len(vec)):
            yvals = [arr[idx] for arr in arrs]
            if not su.findRecursive(yvals, sympy.zoo):
                legends.append(labels[idx])
                ax.plot(tVals, yvals)
            else:
                print("Invalid value for %s" % labels[idx])
        ax.legend(legends)
        ax.set_xlabel("time")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if isPlot:
            plt.show()
