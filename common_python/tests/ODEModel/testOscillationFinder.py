from common_python.ODEModel.oscillationFinder import OscillationFinder, XDict

import matplotlib
import numpy as np
import os
import tellurium as te
import unittest


IGNORE_TEST = False
IS_PLOT = False
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ANTIMONY_FILE = os.path.join(TEST_DIR, "Model_antimony.ant")
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

    def testmkParameters(self):
        if IGNORE_TEST:
            return
        parameterXD = self.kv.mkParameters(self.roadrunner)
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
        dct = self.finder.mkParameters()

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
        parameterXD = XDict.mkParameters(self.roadrunner)
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
        self.finder.plotTime(title="title", isPlot=IS_PLOT, ylim=[0, 10])

    def testSimulate(self):
        if IGNORE_TEST:
            return
        self.finder.simulate()
        self.finder.plotTime(isPlot=IS_PLOT)
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
        # Find a feasible solution
        for _ in range(3):
            feasibleParameterXD = self.finder.find(
                  initialParameterXD=initialParameterXD)
            if feasibleParameterXD is not None:
                break
        if False:
            self.finder.simulate(parameterXD=feasibleParameterXD)
            self.finder.plot()
        self.assertFalse(initialParameterXD.equals(feasibleParameterXD))

    def testPlotTime(self):
        if IGNORE_TEST:
            return
        self.finder.simulate(endTime=200)
        self.finder.plotTime(startTime=1, endTime=5, isPlot=IS_PLOT)
        self.finder.plotTime(startTime=90, endTime=100, isPlot=IS_PLOT)

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
        self.assertIsNotNone(feasibleParameterXD)
        self.assertFalse(initialParameterXD.equals(feasibleParameterXD))

    def testPlotJacobian(self):
        if IGNORE_TEST:
            return
        self.finder.plotJacobian(isPlot=IS_PLOT)
        self.finder.plotJacobian(cbar=False, isPlot=IS_PLOT)

    def testPlotJacobians(self):
        if IGNORE_TEST:
            return
        files = np.repeat(ANTIMONY_FILE, 4)
        self.finder.plotJacobians(files, isPlot=IS_PLOT)

    def testAnalyzeFile(self):
        if IGNORE_TEST:
            return
        OscillationFinder.analyzeFile(ANTIMONY_FILE, numRestart=0, isPlot=IS_PLOT)


if __name__ == '__main__':
    if IS_PLOT:
        matplotlib.use('TkAgg')
    unittest.main()
