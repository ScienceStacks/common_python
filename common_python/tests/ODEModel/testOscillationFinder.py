from common_python.ODEModel.oscillationFinder import OscillationFinder, XDict

import matplotlib
import numpy as np
import os
import tellurium as te
import unittest


IGNORE_TEST = False
IS_PLOT = False
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ANTIMONY_FILE = os.path.join(TEST_DIR, "antimony.ant")
BIG_ANTIMONY_FILE = os.path.join(TEST_DIR, "bigAntimony.ant")
DCT = {'a': 1, 'b': 2}


#############################
# Tests
#############################
class TestXDict(unittest.TestCase):

    def setUp(self):
        self.roadrunner = te.loada(ANTIMONY_FILE)
        self.kv = XDict(names=DCT.keys(), values=DCT.values())

    def testConstructor1(self):
        if IGNORE_TEST:
            return
        self.assertTrue(self.kv.equals(DCT))

    def testConstructor2(self):
        if IGNORE_TEST:
            return
        value = 10
        dct = {k: value for k in DCT.keys()}
        kv = XDict(names=dct.keys(), values=value)
        self.assertFalse(self.kv.equals(kv))
        self.assertTrue(kv.equals(dct))

    def testConstructor3(self):
        if IGNORE_TEST:
            return
        kv = XDict()
        kv1 = XDict()
        self.assertFalse(self.kv.equals(kv))
        self.assertTrue(kv.equals(kv1))

    def testEquals(self):
        if IGNORE_TEST:
            return
        self.assertTrue(self.kv.equals(DCT))
        dct = dict(DCT)
        dct[list(dct.keys())[0]] = -1.0
        self.assertFalse(self.kv.equals(dct))

    def testmkParameter(self):
        if IGNORE_TEST:
            return
        parameterXD = self.kv.mkParameter(self.roadrunner)
        self.assertTrue("k1" in parameterXD.keys())

    def testmkSpecies(self):
        if IGNORE_TEST:
            return
        speciesXD = self.kv.mkSpecies(self.roadrunner)
        self.assertTrue("S1" in speciesXD.keys())
        

class TestOscillationFinder(unittest.TestCase):

    def setUp(self):
        rr = te.loada(ANTIMONY_FILE)
        self.finder = OscillationFinder(rr)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertGreater(len(self.finder.parameterXD), 0)
        self.assertTrue(isinstance(self.finder.parameterXD, dict))

    def testMkParameterXD(self):
        if IGNORE_TEST:
            return
        dct = self.finder.mkParameter()

class TestOscillationFinder(unittest.TestCase):

    def setUp(self):
        self.roadrunner = te.loada(ANTIMONY_FILE)
        self.finder = OscillationFinder(self.roadrunner)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertGreater(len(self.finder.parameterXD), 0)
        self.assertTrue(isinstance(self.finder.parameterXD, dict))

    def mkParameterXD(self, values):
        parameterXD = XDict.mkParameter(self.roadrunner)
        return XDict(parameterXD.keys(), values)

    def testSetSteadyState(self):
        if IGNORE_TEST:
            return
        speciesXD = XDict.mkSpecies(self.roadrunner)
        value = 1
        parameterXD = self.mkParameterXD(value)
        fp = self.finder.setSteadyState(parameterXD=parameterXD)
        self.assertFalse(speciesXD.equals(fp))
        self.assertGreater(fp["S1"], fp["S0"])
        self.assertGreater(fp["S2"], fp["S1"])
        self.finder.plot(isPlot=IS_PLOT)

    def testSimulate(self):
        if IGNORE_TEST:
            return
        self.finder.simulate()
        self.finder.plot(isPlot=IS_PLOT)
        #
        parameterXD = self.mkParameterXD(1)
        self.finder.simulate(parameterXD=parameterXD)
        self.finder.plot(isPlot=IS_PLOT)

    def testGetEigenvalues(self):
        if IGNORE_TEST:
            return
        eigenvalues = self.finder._getEigenvalues()
        self.assertEqual(len(eigenvalues), 3)
        imags = [np.imag(v) for v in eigenvalues if np.imag(v) !=  0]
        self.assertTrue(np.isclose(imags[0], -imags[1]))

    def testFindSmall(self):
        if IGNORE_TEST:
            return
        # Initialize all parameters to 1
        initialParameterXD = self.mkParameterXD(1)
        self.finder.setParameters(initialParameterXD)
        # Find a feasible solution
        feasibleParameterXD = self.finder.find()
        self.assertFalse(initialParameterXD.equals(feasibleParameterXD))

    def testFindBig(self):
        if IGNORE_TEST:
            return
        roadrunner = te.loada(BIG_ANTIMONY_FILE)
        finder = OscillationFinder(roadrunner)
        # Initialize all parameters to 1
        initialParameterXD = self.mkParameterXD(1)
        finder.setParameters(initialParameterXD)
        # Find a feasible solution
        feasibleParameterXD = finder.find()
        self.assertFalse(initialParameterXD.equals(feasibleParameterXD))


if __name__ == '__main__':
    if IS_PLOT:
        matplotlib.use('TkAgg')
    unittest.main()
