"""
A symbol system is a set of equations in which each symbol appears on the
left hand side and the right hand side is an expression in terms of
a subset (possibly empty) of the other symbols.
"""

from common_python.sympy import sympyUtil as su

import copy
import sympy


class SymbolSystem():

    def __init__(self, systemDct):
        """
        Parameters
        ----------
        systemDct: dict
            key: symbol
            value: expression in terms of the other symbols
        """
        self.systemDct = copy.deepcopy(systemDct)
        self.symbols = list(systemDct.keys())

    def _getFreeSymbols(self, epr):
        """
        Finds the symbols in expression that are on the equation LHS.

        Parameters
        ----------
        epr: expression
        
        Returns
        -------
        list-symbol
        """
        freeSymbols = epr.free_symbols
        return [s for s in set(freeSymbols).intersection(self.symbols)]

    def substitute(self, maxNode=50, sequence=None):
        """
        Attempts to simplify the expression by substitution.
        Substitutions are made only if the result is fewer free symbols.
        Updates self.systemDct.

        Parameters
        ----------
        maxNode: int
           maxium complexity of expression (in unints of number of nodes)
        sequence: list-symbol/list-(symbol, symbol)
           sequence in which symbols are used in substitutions
           a tuple is treated as a specific substitution

        Returns
        -------
        dict
            key: symbol
            value: expression
        """
        # TODO: Change the search to minimize the number of distinct symbols
        #       on in the expressions.
        #       Do a substitution if: (a) the symbol is present in the expression
        #                             (b) the RHS for the symbols are a subset of
        #                                 the current distinct symbols.
        def isComplexExpression(sym):
            return su.countNodes(systemDct[sym]) > maxNode
        def doSubstitution(sym1, sym2):
            # Don't substitute a symbol for itself
            if sym1 == sym2:
                return
            # Avoid creating overly complicated expressions
            if isComplexExpression(sym1) or isComplexExpression(sym2):
                return
            # Don't put the symbol in its own expression
            if sym1 in systemDct[sym2].free_symbols:
                return
            systemDct[sym2] = systemDct[sym2].subs(sym1, systemDct[sym1])
        #
        systemDct = dict(self.systemDct)
        # Set the order in which symbols are used for substitutions
        if sequence is None:
            sortedSymbols = sorted(self.symbols,
                  key=lambda s: len(systemDct[s].free_symbols))
        else:
            sortedSymbols = sequence
        # Do the substitutions
        for _ in range(len(systemDct.keys())):
            for entry in sortedSymbols:
                if isinstance(entry, tuple):
                    sym1 = entry[0]
                    sym2 = entry[1]
                else:
                    sym1 = entry
                    sym2 = None
                distinctSymbols = su.getDistinctSymbols(systemDct)
                currentSymbols = su.getDctSymbols(systemDct)[sym1]
                if not sym1 in distinctSymbols:
                    continue
                if not set(currentSymbols).issubset(distinctSymbols):
                    continue
                # Use first expressions with fewer free symbols
                if sym2 is not None:
                    doSubstitution(sym1, sym2)
                else:
                    [doSubstitution(sym1, s) for s in systemDct.keys()]
        # Force substitutions for symbols with expressions that don't
        # have other symbols
        for _ in range(len(systemDct)):
            done = True
            countDct = su.countDctSymbols(systemDct)
            for sym, count in countDct.items():
                if count == 0:
                    done = False
                    systemDct = {s: systemDct[s].subs(sym, systemDct[sym])
                          for s in systemDct.keys()}
            if done:
                break
        #
        return systemDct

    @staticmethod
    def _combineLists(lsts):
        """
        Creates lists that are all combinations of the elements of a list of lists.
        Consider
           lsts = [ [ 1, 2, 3], [4, 5]]
        This method returns
           [ [1, 4], [1, 5], [2, 4], [2, 5], [3, 4], [3, 5] ]

        Parameters
        ----------
        lsts: list-list
        
        Returns
        -------
        list-list
        """
        # If this is the last list, just return it
        if len(lsts) == 1:
            return [ [e] for e in lsts[0]]
        # Otherwise, recursive construct the lists
        tailLsts = SymbolSystem._combineLists(lsts[1:])
        resultLsts = []
        for ele in lsts[0]:
            for lst in tailLsts:
                newLst = list(lst)
                newLst.insert(0, ele)
                resultLsts.append(newLst)
        return resultLsts
                
    @classmethod
    def mkSymbolSystem(cls, eqnDct):
        """
        Creates symbol systems from expressions that may contain the
        the desired symbol.

        Parameters
        ----------
        eqnDct: dict
            key: symbol
            value: expression
        
        Returns
        -------
        list-SymbolSystem
        """
        dct = {}
        for sym, epr in eqnDct.items():
            if sym in epr.free_symbols:
                dct[sym] = sympy.solve(epr, [sym])
            else:
                dct[sym] = [epr]
        # Remove solutions that have infinities
        for sym, expressions in dct.items():
            dct[sym] = [e for e in expressions 
                  if not e.has(sympy.oo, -sympy.oo, sympy.zoo, sympy.nan)]
        # Ensure non-null lists
        dct = {s: [None] if len(v) == None else v for s, v in dct.items()}
        # Construct all combinations of solutions
        lsts = cls._combineLists(list(dct.values()))
        dcts = [ {k: l for k,l in zip(dct.keys(), l)} for l in lsts]
        systems = [cls(d) for d in dcts]
        return systems
