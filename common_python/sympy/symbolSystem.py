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
            value: expression
        """
        self.systemDct = copy.deepcopy(systemDct)
        self.symbols = list(systemDct.keys())

    @staticmethod
    def _getFreeSymbols(epr):
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
        return [s for s in set(freesymbols).intersection(self.symbols)]

    def substitute(self, symbolsWhoseExpressionsAreSubstituted=None,
              symbolsToBeSubstituted=None):
        """
        Attempts to simplify the expression by substitution.
        Substitutions are made only if the result is fewer free symbols.
        Updates self.systemDct.

        Parameters
        ----------
        symbolsWhoseExpressionsAreSubstituted: list-symbol
        symbolsToBeSubstituted: list-symbol
        """
        if symbolsWhoseExpressionsAreSubstituted is None:
            symbolsWhoseExpressionsAreSubstituted = self.symbols
        if symbolsToBeSubstituted is None:
            symbolsToBeSubstituted = self.symbols
        #
        fsortedSymbols = sorted(sortedSymbols, key=lambda s: len(s.free_symbols))
        bsortedSymbols = list(fsortedSymbols)
        bsortedSymbols.reverse()
        # Substitute first into expressions with more symbols
        systemDct = dict(self.systemDct)
        for sym1 in bsortedSymbols:
            if not sym1 in symbolsWHoseExpressionsAreSubstituted:
                continue
            # Use first expressions with fewer free symbols
            for sym2 in fsortedSymbols:
                if sym1 == sym2:
                    continue
                if not sym2 in symbolsToBeSubstituted:
                    continue
                symbols1 = self._getFreeSymbols(self.systemDct[sym1])
                symbols2 = self._getFreeSymbols(self.systemDct[sym2])
                commonSymbols = set(symbols2).intersection(symbols1)
                # Only substitute if not increasing the number of symbols
                if len(symbols2) - len(commonSymbols) <= 1:
                    systemDct[sym1] = self.systemDct[sym1].subs(sym2,
                          self.systemDct[sym2])

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
        dct = {s: sympy.solve(e, [s]) for k, s in eqnDct.items()}
        # Ensure non-null lists
        dct = {s: [None] if len(v) == None else v for k, v in dct.items()}
        # Construct all combinations of solutions
        lsts = list(dct.values())
        dcts = [ {k: l for k,l in zip(dct.keys(), l)} for l in lsts]
        systems = [cls(d) for d in dcts]
        return systems
