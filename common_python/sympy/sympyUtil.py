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

import copy
import inspect
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

def addSymbols(symbolStr, dct=None, real=True, negative=False):
    """
    Adds symbols to the dictionary.

    Parameters
    ----------
    symbolStr: str
    dct: dict
        default: globals() of caller
    kwarg: optional arguments for sympy.symbols
    """
    newDct = _getDct(dct, inspect.currentframe())
    _addSymbols(symbolStr, newDct, real=real, negative=negative)

def _addSymbols(symbolStr, dct, **kwargs):
    symbols = symbolStr.split(" ")
    for symbol in symbols:
        dct[symbol] = sympy.Symbol(symbol, **kwargs)

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

def isSymbol(obj):
    if "is_symbol" in dir(obj):
        return obj.is_symbol
    else:
        return False

def substitute(expression, subs=None):
    """
    Substitutes into the expression.

    Parameters
    ----------
    expression: sympy.expression
    subs: dict
       key: sympy.symbol
       value: number

    Returns
    -------
    sympy.expression
    """
    if subs is None:
        subs = {}
    if isNumber(expression):
        return expression
    if isSymbol(expression):
        if expression.name in subs:
            return subs[expression.name]
        elif expression in subs:
            return subs[expression]
        else:
            return expression
    expr = expression.copy()
    # Must be an expression
    symbolDct = {s.name: s for s in expression.free_symbols}
    # Update entry in substitution to be the same as the expression
    newSubs = dict(subs)
    for key, value in subs.items():
        if key.name in symbolDct.keys():
            del newSubs[key]
            newSubs[symbolDct[key.name]] = value
    expr = expr.subs(newSubs)
    return sympy.simplify(expr)

def evaluate(expression, isNumpy=True, **kwargs):
    """
    Evaluates the solution for the substitutions provided.

    Parameters
    ----------
    expression: sympy.Add/number
    isNumpy: bool
        return float or ndarray of float
    kwargs: dict
        keyword arguments for substitute

    Returns
    -------
    float/np.ndarray
    """
    if isSympy(expression):
        val = _evaluate(expression, isNumpy=isNumpy, **kwargs)
    else:
        val = expression
    return val

def _evaluate(expression, isNumpy=True, **kwargs):
    """
    Evaluates the solution for the substitutions provided.

    Parameters
    ----------
    expression: sympy.Add
    isNumpy: bool
        return float or ndarray of float
    kwargs: dict
        keyword arguments for substitute

    Returns
    -------
    float/np.ndarray
    """
    if isNumber(expression):
        if isNumpy:
            return expressionToNumber(expression)
        else:
            return expression
    # Evaluate
    expr = substitute(expression, **kwargs)
    # Symbol substitution can create a number
    if isNumber(expr):
        return expr
    val = expr.evalf()
    if hasSymbols(val):
        return val
    if isNumpy:
        if "rows" in dir(expression):
            result = np.array(val)
        else:
            try:
                result = float(val)
            except TypeError:
                result = complex(val)
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
    return evaluate(sympy.Matrix(lst), subs=subs)

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
        if val.is_real:
            val = float(val)
        else:
            val = complex(val)
    else:
        val = expression  # already a number
    # Eliminate small values
    if np.abs(val) < SMALL_VALUE:
        val = 0
    if np.abs(np.angle(val)) < SMALL_VALUE:
        val = np.sign(val) * np.abs(val)
    return val

def asRealImag(val):
    if isSympy(val):
        return val.as_real_imag()
    cmplxVal = complex(val)
    return (cmplxVal.real, cmplxVal.imag)

def isConjugate(val1, val2):
    """
    Tests for complex conjugates.

    Parameters
    ----------
    val1: number or expression
    val2: number or expression

    Returns
    -------
    bool
    """
    realImag = [asRealImag(val1), asRealImag(val2)]
    isSameReal = realImag[0][0] == realImag[1][0]
    isSameImag = realImag[0][1] == -realImag[1][1]
    return isSameReal and isSameImag

def _hasSymbols(val):
    if isSympy(val):
        return len(val.free_symbols) > 0
    else:
        return False

def hasSymbols(val):
    if isIndexable(val):
        trues = [_hasSymbols(v) for v in val]
    else:
        trues = [_hasSymbols(val)]
    return any(trues)

def isSympy(val):
    """
    Tests if this is a sympy object.

    Parameters
    ----------
    val: object

    Returns
    -------
    bool
    """
    properties = dir(val)
    return ("is_symbol" in properties) or ("evalf" in properties)

def isNumber(val):
    """
    Tests if this is a number in base python or sympy.

    Parameters
    ----------
    val: float/int/complex/sympy.expression

    Returns
    -------
    bool
    """
    try:
        _ = complex(val)
        return True
    except TypeError:
        return False

def isReal(val):
    if isSympy(val):
        return val.is_real
    return np.isreal(val)

def isComplex(val):
    return isNumber(val) and (not isReal(val))

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
        except Exception:
            return False
    try:
        if np.isclose(np.abs(val), 0):
            return True
    except TypeError:
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
    # FIXME: find the symbols using free_symbols
    solution = aMat.gauss_jordan_solve(bVec)
    solutionVec = solution[0]
    if not isParameterized:
        parameterMat = solution[1]
        for parameter in parameterMat:
            solutionVec = solutionVec.subs(parameter, DEFAULT_VALUE)
    solutionVec = solutionVec.evalf()
    return solutionVec

def isIndexable(obj):
    return "__getitem__" in dir(obj)

def recursiveEvaluate(obj, **kwargs):
    """
    Recursively evaluates symbols encountered in muteable indexables of type:
    list, np.array, sympy.Matrix (and maybe others)

    Parameters
    ----------
    obj: sympy.expr/number/indexable
        an indexable support indexing.
    kwargs: dict
        keyword arguments for evaluate

    Returns
    -------
    object with same structure as original
    """
    if isIndexable(obj):
        # Is a container of other objects
        if "tuple" in str(obj.__class__):
            newObj = list(obj)
        else:
            newObj = copy.deepcopy(obj)
        newerObj = copy.deepcopy(newObj)
        for idx, entry in enumerate(newObj):
            newerObj[idx] = recursiveEvaluate(entry, **kwargs)
        return newerObj
    # Do the numeric evaluation
    return evaluate(obj, **kwargs)

def recursiveEquals(obj1, obj2, **kwargs):
    """
    Recursively evaluates if two indexable, mutable objects are equal
    under subtituions.

    Parameters
    ----------
    obj: sympy.expr/number/indexable
        an indexable support indexing.
    kwargs: dict
        keyword arguments for evaluate

    Returns
    -------
    bool
    """
    if isIndexable(obj1) != isIndexable(obj2):
        return False
    if isIndexable(obj1):
        for entry1, entry2 in zip(obj1, obj2):
            if not recursiveEquals(entry1, entry2, **kwargs):
                return False
        return True
    # Do the numeric evaluation
    num1 = expressionToNumber(evaluate(obj1, **kwargs))
    num2 = expressionToNumber(evaluate(obj2, **kwargs))
    return np.isclose(num1, num2)

def vectorAsRealImag(vec):
    """
    Expresses a vector as the sum of real and imaginary parts.

    Parameters
    ----------
    vec - sympy.Matrix

    Returns
    -------
    sympy.Matrix, sympy.Matrix
        real vector, imaginary vector
    """
    reals = []
    imags = []
    for entry in vec:
        real, imag = asRealImag(entry)
        reals.append(real)
        imags.append(imag)
    return sympy.Matrix(reals),  sympy.Matrix(imags)

def eigenvects(mat):
    """
    Computes eigenvector results, handling near zero determinants.

    Parameters
    ----------
    mat: sympy.Matrix
    
    Returns
    -------
    eigenvect entries returned by sympy.eigenvects()
    """
    # Check if symbols are present
    if hasSymbols(mat):
        return mat.eigenvects()
    # Purely numeric matrix
    newMat = recursiveEvaluate(mat.as_mutable())
    return newMat.eigenvects()
