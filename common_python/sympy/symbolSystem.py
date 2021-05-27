"""
A collection of symbols paired with an expression for the symbol.
The symbol does not appear in the expression.
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

    def do(self, symbolsWhoseExpressionsAreSubstituted=None,
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
        for sym1 in symbols:
            if not sym1 in symbolsWHoseExpressionsAreSubstituted:
                continue
            for sym2 in symbols:
                if sym1 == sym2:
                    continue
                if not sym2 in symbolsToBeSubstituted:
                    continue
                symbols1 = self._getFreeSymbols(self.systemDct[sym1])
                symbols2 = self._getFreeSymbols(self.systemDct[sym2])
                if set(symbols2).issubset(symbols1):
                    self.systemDct[sym1] = self.systemDct[sym1].subs(sym2,
                          self.systemDct[sym2])

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
        # TODO: Construct all combinations of solutions
