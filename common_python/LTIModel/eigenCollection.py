"""Collects information for eigenvalue and vectors.
Key properties:
    value - eigenvalue
    vectors - eigenvectors
    algebraicMultipliciaty - algebraic multiplicity
""" 

import common_python.LTIModel.constants as cn
import common_python.sympy.sympyUtil as su

import collections
import numpy as np
import sympy

SMALL_VALUE = 1e-8
t = sympy.Symbol(cn.SYM_T)


class EigenInfo():
    # Information about one eigenvalue and its eigenvectors

    def __init__(self, matrix, val, vecs, mul):
        """
        Parameters
        ----------
        matrix: sympy.Matrix N X N
        val: float
            Eigenvalue
        vecs: list-sympy.Matrix
            eigenvectors
        mul: int
            algebraic multiplicity
        """
        self.matrix = matrix
        self.numRow = self.matrix.rows
        self.val = val
        self.vecs = vecs
        self.mul = mul

    def copy(self):
        return EigenInfo(self.matrix.copy(), self.val, self.vecs.copy(),
              self.mul)

    # TODO: Not sure that I'm using the correct vectors for those added
    def completeEigenvectors(self):
        """
        Adds eigenVectors if the algebraic multiciplicity > geometric multiciplicy
        Updates self.vecs
        """
        if self.mul == len(self.vecs):
            return
        lastVec = self.vecs[0]
        newVecs = []  # Constructed solution vectors
        termVecs = [lastVec]  # vectors used in constructing solution vectors
        numvec = len(self.vecs)
        for num in range(1, numvec+1):
            # Compute a new vector
            mat = self.matrix - sympy.eye(self.numRow) * self.val
            termVec = su.solveLinearSingular(mat, lastVec)
            termVecs.append(termVec)
            # Construct the vector that is a function of time
            for idx in range(len(termVecs) - 1 ):
                termIdx = idx + 1
                lastVec = termVecs[idx]
                curVec = termVecs[termIdx]
                timeVec = curVec + t**termIdx / sympy.factorial(termIdx)  \
                      * lastVec
            newVecs.append(timeVec)
            lastVec = timeVec
        self.vecs.extend(newVecs)
        
    
class EigenCollection():
    # Container for all EigenInfo for a matrix

    def __init__(self, matrix):
        """
        Parameters
        ----------
        matrix: sympy.Matrix N X N
        """
        def simplify(v):
            newV = self._roundToZero(v)
            try:
                finalV = su.expressionToNumber(newV)
            except TypeError:
                finalV = newV  # Cannot convert an experssion
            return finalV
        #
        self.matrix = matrix
        eigenInfos = []  # Container for eigenInfos
        self.eigenvalDct = {simplify(k): v for k, v in 
              self.matrix.eigenvals().items()}
        # Create the raw EigenInfo
        for entry in self.matrix.eigenvects():
            eigenvalue = simplify(entry[0])
            algebraicMultiplicity = self.eigenvalDct[eigenvalue]
            vecs = [self._vectorRoundToZero(v) for v in entry[2]]
            vecs = [v.evalf() for v in vecs]
            eigenInfos.append(EigenInfo(
                  self.matrix,
                  eigenvalue,
                  vecs,
                  algebraicMultiplicity))
        # Prune: Sort by Eigenvalue. Then do pairwise _merge.
        self.eigenInfos = eigenInfos
        try:
            self.eigenInfos = sorted(eigenInfos, key=lambda e: np.abs(e.val))
            eigenInfos = [self.eigenInfos[0]]
            for idx, eigenInfo in enumerate(self.eigenInfos[:-1]):
                otherEigenInfo = self.eigenInfos[idx+1]
                if np.isclose(eigenInfo.val, otherEigenInfo.val):
                    # TODO: Should I look combine the vecs in otherEigenInfo?
                    pass
                else:
                    eigenInfos.append(otherEigenInfo)
            numEigenValue = sum([e.mul for e in eigenInfos])
            if numEigenValue != self.matrix.rows:
                raise RuntimeError("Missing or extra eigenvalue?")
            self.eigenInfos = eigenInfos
        except TypeError:
            pass  # Cannot prune if a symbol is present

    # TODO: Is this needed?
    @staticmethod
    def eliminateDuplicateVectors(vecs):
        """
        Eliminates vectors that have the same values.

        Parameters
        ----------
        vecs: list-sympy.Matrix
        
        Returns
        -------
        vecs: list-sympy.Matrix
        """
        results = []
        for idx in range(len(vecs) - 1):
            curVec = vecs[idx]
            results.append(curVec)  # Assume it differs from other vectors
            for vec in vecs[idx+1:]:
                if curVec.rows != vec.rows:
                    continue
                if su.isVecZero(curVec - vec):
                    _ = results.pop()  # Remove curVec
                    break

    def completeEigenvectors(self):
        """
        Ensures that all eigenvalues have a complete set of eigenvectors.
        """
        _ = [e.completeEigenvectors() for e in self.eigenInfos]
   
    @staticmethod 
    def _vectorRoundToZero(vec):
        if vec.cols > 1:
            RuntimeError("Can only handle vectors.")
        newValues = [EigenCollection._roundToZero(v) for v in vec]
        return sympy.Matrix(newValues)
    
    @staticmethod 
    def _roundToZero(v):
        if "is_symbol" in dir(v):
            if not v.is_Number:
                return v
        if np.abs(v) < SMALL_VALUE:
            return 0
        return v

    def pruneConjugates(self):
        """
        Returns the eigenInfos with only one member of each conjugate pair.
        
        Returns
        -------
        list-EigenInfo
        """
        eigenInfos = []
        for idx, eigenInfo1 in enumerate(self.eigenInfos[:-1]):
            isConjugate = False
            for eigenInfo2 in self.eigenInfos[idx + 1:]:
                if su.isConjugate(eigenInfo1.val, eigenInfo2.val):
                    isConjugate = True
                    break
            if not isConjugate:
                eigenInfos.append(eigenInfo1)
        eigenInfos.append(self.eigenInfos[-1])
        return eigenInfos
            
            
