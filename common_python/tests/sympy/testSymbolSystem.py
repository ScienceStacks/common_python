from common_python.sympy.symbolSystem import SymbolSystem
import common_python.sympy.sympyUtil as su
import common_python.sympy.constants as cn

import sympy
import unittest


IGNORE_TEST = False
IS_PLOT = False
STATE_VARIABLES = "Ach Adp Atp Fru Ghp Glu Gt3p Nad Nadh Pyr"
su.addSymbols(STATE_VARIABLES, dct=globals())
CONSTANTS = "J0 J1_k1 J1_Ki J1_n J2_k J3_k J4_kg J4_kp J4_k1 J4_kk J4_ka "
CONSTANTS += " J5_k J6_k J7_k J8_k1 J8_k2 J9_k Eah"
su.addSymbols(CONSTANTS, dct=globals())
systemDct = {Ach: Ach*J7_k*Nadh + Ach*J8_k1 - Eah*J8_k2 - J6_k*Pyr,
      Adp: -Adp*Gt3p*J5_k + 2*Atp*Glu*J1_k1/((Atp/J1_Ki)**J1_n
       + 1) + Atp*J9_k - (Adp*Ghp*J4_kg*J4_kp*Nad - Atp*Gt3p*J4_ka*J4_kk*Nadh)
       /(Adp*J4_kp + J4_ka*Nadh),
      Atp: Adp*Gt3p*J5_k - 2*Atp*Glu*J1_k1/((Atp/J1_Ki)**J1_n + 1) - Atp*J9_k + (Adp*Ghp*J4_kg*J4_kp*Nad - Atp*Gt3p*J4_ka*J4_kk*Nadh)/(Adp*J4_kp + J4_ka*Nadh),
      Fru: Atp*Glu*J1_k1/((Atp/J1_Ki)**J1_n + 1) - Fru*J2_k,
      Ghp: -2*Fru*J2_k + Ghp*J3_k*Nadh - (Adp*Ghp*J4_kg*J4_kp*Nad - Atp*Gt3p*J4_ka*J4_kk*Nadh)/(Adp*J4_kp + J4_ka*Nadh),
      Glu: -Atp*Glu*J1_k1/((Atp/J1_Ki)**J1_n + 1) + J0,
      Gt3p: -Adp*Gt3p*J5_k + (Adp*Ghp*J4_kg*J4_kp*Nad - Atp*Gt3p*J4_ka*J4_kk*Nadh)/(Adp*J4_kp + J4_ka*Nadh),
      Nad: Ach*J7_k*Nadh + Ghp*J3_k*Nadh - (Adp*Ghp*J4_kg*J4_kp*Nad - Atp*Gt3p*J4_ka*J4_kk*Nadh)/(Adp*J4_kp + J4_ka*Nadh),
      Nadh: -Ach*J7_k*Nadh - Ghp*J3_k*Nadh + (Adp*Ghp*J4_kg*J4_kp*Nad - Atp*Gt3p*J4_ka*J4_kk*Nadh)/(Adp*J4_kp + J4_ka*Nadh),
      Pyr: Adp*Gt3p*J5_k - J6_k*Pyr}




#############################
# Tests
#############################
class TestSymbolSystem(unittest.TestCase):

    def setUp(self):
        self.system = SymbolSystem(systemDct)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertGreater(len(self.system.symbols), 0)



if __name__ == '__main__':
  unittest.main()
