""" Utilities for sympy. ***

SUffix conventions
  *Arr - numpy array
  *Mat - sympy.Matrix N X N
  *Vec - sympy.Matrix N X 1
  *s - list

Notes
1. By default, symbols are added to locals() of the caller.
   This can be changed by using the dct optional keyword.
   This means that all user callable functions must capture
   the correct symbol dictionary and use this explicitly
   in internal calls.

"""

import inspect
import matplotlib.pyplot as plt
import numpy as np
import sympy

SMALL_VALUE = 1e-8
DEFAULT_VALUE = 1.0  # Value substituted in parameterized expressions

def _getDct(dct, frame):
    """
    Gets the dictionary for the frame.

    Parameters
    ----------
    dct: dictionary to use if non-None
    frame: stack frame
    
    Returns
    -------
    dict
    """
    if dct is None:
        dct = frame.f_back.f_locals
    return dct

def addSymbols(symbolStr, dct=None):
    """
    Adds symbols to the dictionary.

    Parameters
    ----------
    symbolStr: str
    dct: dict
        default: globals() of caller
    """
    newDct = _getDct(dct, inspect.currentframe())
    _addSymbols(symbolStr, newDct)

def _addSymbols(symbolStr, dct):
    symbols = symbolStr.split(" ")
    for idx, symbol in enumerate(symbols):
        dct[symbol] = sympy.Symbol(symbol)

def removeSymbols(symbolStr, dct=None):
    """
    Removes symbols from the dictionary.

    Parameters
    ----------
    symbolStr: str
    dct: dict
        Namespace dictionary
    """
    newDct = _getDct(dct, inspect.currentframe())
    _removeSymbols(symbolStr, newDct)

def _removeSymbols(symbolStr, dct):
    symbols = symbolStr.split(" ")
    for symbol in symbols:
        if symbol in dct.keys():
            del dct[symbol]

def substitute(expression, subs={}):
    """
    Substitutes into the expression.

    Parameters
    ----------
    subs: dict
       key: sympy.symbol
       value: number
    
    Returns
    -------
    sympy.expression
    """
    expr = expression.copy()
    for key, value in subs.items():
        expr = expr.subs(key, value)
    return sympy.simplify(expr)

def evaluate(expression, dct=None, isNumpy=True, **kwargs):
    """
    Evaluates the solution for the substitutions provided.

    Parameters
    ----------
    expression: sympy.Add
    isNumpy: bool
        return float or ndarray of float
    dct: dict
        Namespace dictionary
    kwargs: dict
        keyword arguments for substitute
    
    Returns
    -------
    float/np.ndarray
    """
    newDct = _getDct(dct, inspect.currentframe())
    return _evaluate(expression, isNumpy=isNumpy, dct=newDct, **kwargs)

def _evaluate(expression, dct, isNumpy=True, **kwargs):
    """
    Evaluates the solution for the substitutions provided.

    Parameters
    ----------
    expression: sympy.Add
    isNumpy: bool
        return float or ndarray of float
    dct: dict
        Namespace dictionary
    kwargs: dict
        keyword arguments for substitute
    
    Returns
    -------
    float/np.ndarray
    """
    expr = substitute(expression, **kwargs)
    val = expr.evalf()
    if isNumpy:
        if "rows" in dir(expression):
            result = np.array(val)
        else:
            try:
                result = float(val)
            except Exception:
                import pdb; pdb.set_trace()
                pass
    else:
        result = val
    return result

def mkVector(nameRoot, numRow, dct=None):
    """
    Constructs a vector of symbols.

    Parameters
    ----------
    nameRoot: str
        root name for elements of the vector
    numRow: int
    dct: dict
        Namespace dictionary
    
    Returns
    -------
    sympy.Matrix numRow X 1
    """
    newDct = _getDct(dct, inspect.currentframe())
    return _mkVector(nameRoot, numRow, newDct)

def _mkVector(nameRoot, numRow, dct):
    # Create the solution vector. The resulting vector is in the global name space.
    symbols = ["%s_%d" % (nameRoot, n) for n in range(numRow)]
    symbolStr = " ".join(symbols)
    addSymbols(symbolStr, dct=dct)
    return sympy.Matrix([ [s] for s in symbols])

def flatten(vec):
    """
    Converts a sympy N X 1 matrix to a list.

    Parameters
    ----------
    vec: symbpy.Matrix N X 1
    
    Returns
    -------
    list
    """
    return [ [v for v in z] for z in vec][0]
    
def vectorRoundToZero(vec):
    if vec.cols > 1:
        RuntimeError("Can only handle vectors.")
    newValues = [roundToZero(v) for v in vec]
    return sympy.Matrix(newValues)

def roundToZero(v):
    if isSympy(v):
        if not v.is_Number:
            return v
    if np.abs(v) < SMALL_VALUE:
        return 0
    return v

def solveLinearSystem(aMat, bMat):
    """
    Finds a solution to A*x = b. Chooses an arbitrary
    solution if multiple solutions exist.

    Parameters
    ----------
    aMat: sympy.Matrix (N X N)
        A
    bMat: sympy.Matrix (N X 1)
        b
    
    Returns
    -------
    sympy.Matrix: N X 1
    """
    numRow = aMat.rows
    dummyVec = mkVector("x", numRow)
    dummySymbols = [v for v in dummyVec]
    #
    system = aMat, bMat
    result = sympy.linsolve(system, *dummyVec)
    lst = flatten(result)
    # Handle case of multiple solutions
    subs = {s: 1 for s in lst if s in dummySymbols}
    return sympy.Matrix(lst)

def expressionToNumber(expression):
    """
    Converts an exprssion to a numpy number.
    Throws an exception if it cannot be done.

    Parameters
    ----------
    expression: sympy.Add
    
    Returns
    -------
    float/complex
    
    Raises
    -------
    TypeError if not convertable to a number
    """
    # Convert expression to a number
    if isSympy(expression):
        val = expression.evalf()
        try:
            val = float(val)
        except TypeError:
            val = complex(val)
    else:
        val = expression  # already a number
    # Eliminate small values
    if np.abs(val) < SMALL_VALUE:
        val = 0
    if np.angle(val) < SMALL_VALUE:
        val = np.sign(val) * np.abs(val)
    return val

def isSympy(val):
    properties = dir(val)
    return ("is_symbol" in properties) or ("evalf" in properties)
    
def isZero(val):
    """
    Tests if a scalar, number or symbol, is 0.
   
    Parameters
    ----------
    val: number or symbol or expression
    
    Returns
    -------
    bool
    """
    if isSympy(val):
        try:
            val = expressionToNumber(val)
        except:
            return False
    try:
        if np.isclose(np.abs(val), 0):
            return True
    except TypeError:
        import pdb; pdb.set_trace()
        newVal = complex(val)
        return np.abs(newVal) == 0

def isVecZero(vec):
    """
    Tests if a vector, numbers or symbols, are equal.
   
    Parameters
    ----------
    vec: sympy.Matrix (N X 1)

    Returns
    -------
    bool
    """
    trues = [isZero(e) for e in vec]
    return all(trues)

def solveLinearSingular(aMat, bVec, isParameterized=False):
    """
    Solves a linear system where the matrix may be singular.
    Parameter values are set to one if not isParameterized.

    Parameters
    ----------
    aMat: sympy.Matrix N X N
    bVec: sympy.Matrix N X 1
    
    Returns
    -------
    sympy.Matrix N X 1
    """
    solution = aMat.gauss_jordan_solve(bVec)
    solutionVec = solution[0]
    if not isParameterized:
        parameterMat = solution[1]
        for parameter in parameterMat:
            solutionVec = solutionVec.subs(parameter, DEFAULT_VALUE)
    solutionVec = solutionVec.evalf()
    return solutionVec
   
    
    
