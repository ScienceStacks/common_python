from common_python.sympy.symbolSystem import SymbolSystem
import common_python.sympy.sympyUtil as su
import common_python.sympy.constants as cn

import numpy as np
import sympy
import unittest


IGNORE_TEST = True
IS_PLOT = True
STATE_VARIABLES = "Ach Adp Atp Fru Ghp Glu Gt3p Nad Nadh Pyr"
su.addSymbols(STATE_VARIABLES, dct=globals())
CONSTANTS = "J0 J1_k1 J1_Ki J1_n J2_k J3_k J4_kg J4_kp J4_k1 J4_kk J4_ka "
CONSTANTS += " J5_k J6_k J7_k J8_k1 J8_k2 J9_k Eah"
su.addSymbols(CONSTANTS, dct=globals())
su.addSymbols("X Y Z", dct=globals())
EXPR_VALUE = 3
EXPR = EXPR_VALUE * X
EXPR = EXPR.subs(X, 1)
# All expressions have EXPR_VALUE
SIMPLE_SYSTEM_DCT = {Ach: Atp, Adp: 2 * Ach - Atp, Atp: EXPR}
# Complicated system
SYSTEM_DCT = {
      Ach: Ach*J7_k*Nadh + Ach*J8_k1 - Eah*J8_k2 - J6_k*Pyr,
      Adp: -Adp*Gt3p*J5_k + 2*Atp*Glu*J1_k1/((Atp/J1_Ki)**J1_n
        + 1) + Atp*J9_k - (Adp*Ghp*J4_kg*J4_kp*Nad - Atp*Gt3p*J4_ka*J4_kk*Nadh)
        /(Adp*J4_kp + J4_ka*Nadh),
      Atp: Adp*Gt3p*J5_k - 2*Atp*Glu*J1_k1/((Atp/J1_Ki)**J1_n + 1) - Atp*J9_k
        + (Adp*Ghp*J4_kg*J4_kp*Nad - Atp*Gt3p*J4_ka*J4_kk*Nadh)/(Adp*J4_kp
        + J4_ka*Nadh),
      Fru: Atp*Glu*J1_k1/((Atp/J1_Ki)**J1_n + 1) - Fru*J2_k,
      Ghp: -2*Fru*J2_k + Ghp*J3_k*Nadh - (Adp*Ghp*J4_kg*J4_kp*Nad
        - Atp*Gt3p*J4_ka*J4_kk*Nadh)/(Adp*J4_kp + J4_ka*Nadh),
      Glu: -Atp*Glu*J1_k1/((Atp/J1_Ki)**J1_n + 1) + J0,
      Gt3p: -Adp*Gt3p*J5_k + (Adp*Ghp*J4_kg*J4_kp*Nad
        - Atp*Gt3p*J4_ka*J4_kk*Nadh)/(Adp*J4_kp + J4_ka*Nadh),
      Nad: Ach*J7_k*Nadh + Ghp*J3_k*Nadh - (Adp*Ghp*J4_kg*J4_kp*Nad
        - Atp*Gt3p*J4_ka*J4_kk*Nadh)/(Adp*J4_kp + J4_ka*Nadh),
      Nadh: -Ach*J7_k*Nadh - Ghp*J3_k*Nadh + (Adp*Ghp*J4_kg*J4_kp*Nad
        - Atp*Gt3p*J4_ka*J4_kk*Nadh)/(Adp*J4_kp + J4_ka*Nadh),
      Pyr: Adp*Gt3p*J5_k - J6_k*Pyr,
      }




#############################
# Tests
#############################
class TestSymbolSystem(unittest.TestCase):

    def setUp(self):
        self.system = SymbolSystem(SIMPLE_SYSTEM_DCT)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertGreater(len(self.system.symbols), 0)

    def testCombineLists(self):
        if IGNORE_TEST:
            return
        lsts = [ [1, 2, 3], [20], [30, 40]]
        resultLsts = self.system._combineLists(lsts)
        size = np.prod([len(l) for l in lsts])
        self.assertEqual(size, len(resultLsts))
        trues = [len(l) == len(lsts) for l in resultLsts]
        self.assertTrue(all(trues))

    def mkComplicatedSystemDct(self, includes=None, excludes=[Atp, Adp]):
        dct = {k: v for k, v in SYSTEM_DCT.items() if not k in excludes}
        if includes is None:
            return dct
        return {k: v for k, v in dct.items() if k in includes}

    def testMkSymbolSystem(self):
        if IGNORE_TEST:
            return
        systemDct = self.mkComplicatedSystemDct()
        systems = SymbolSystem.mkSymbolSystem(systemDct)
        self.assertEqual(len(systems), 2)
        for system in systems:
            self.assertTrue(isinstance(system, SymbolSystem))

    def testSubstitute1(self):
        if IGNORE_TEST:
            return
        systemDct = self.system.substitute()
        for value in systemDct.values():
            self.assertEqual(value, EXPR_VALUE)

    def testSubstitute2(self):
        if IGNORE_TEST:
            return
        symbols = [Ach, Glu, Atp]
        systemDct = self.mkComplicatedSystemDct(includes=symbols, excludes=[])
        systemDct[Atp] = EXPR
        system = SymbolSystem.mkSymbolSystem(systemDct)[0]
        newSystemDct = system.substitute()
        for sym, epr in newSystemDct.items():
            falses = [s in epr.free_symbols for s in symbols]
            self.assertFalse(any(falses))

    def testSubstitute3(self):
        if IGNORE_TEST:
            return
        systemDct = self.mkComplicatedSystemDct()
        system = SymbolSystem.mkSymbolSystem(systemDct)[0]
        newSystemDct = system.substitute()
        report1 = su.countDctSymbols(system.systemDct, excludes=[Atp, Adp])
        report2 = su.countDctSymbols(newSystemDct, excludes=[Atp, Adp])
        report3 = su.getDctSymbols(newSystemDct, excludes=[Atp, Adp])
        import pdb; pdb.set_trace()

    def testSubstitute4(self):
        # TESTING
        systemDct = self.mkComplicatedSystemDct()
        system = SymbolSystem.mkSymbolSystem(systemDct)[0]
        sequence = [Pyr, Glu, Fru, Ach, Nad, Gt3p, Nadh, Ghp]
        # sequence.extend([(Ghp, Fru), (Nadh, Nad)])
        newSystemDct = system.substitute(sequence=sequence)
        report3 = su.getDctSymbols(newSystemDct)
        distinctSymbols = su.getDistinctSymbols(newSystemDct)
        distinctSymbols = su.getDistinctSymbols(newSystemDct,
              excludes=[Nadh, Pyr, Ghp, Nad])
        import pdb; pdb.set_trace()

    def testCountNodes(self):
        if IGNORE_TEST:
            return
        epr = Ach + (Ach * Glu * (Atp + 3*Ach))
        numNode = su.countNodes(epr)
        self.assertEqual(numNode, 10)
        #
        epr = Ach + Glu
        numNode = su.countNodes(epr)
        self.assertEqual(numNode, 3)


if __name__ == '__main__':
  unittest.main()
