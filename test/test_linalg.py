import unittest
from gpmultipy import linalg
import numpy as np

class TestLinalg(unittest.TestCase):

    def setUp(self,):
        self.L = np.zeros((50,50))

        for i in range(self.L.shape[0]):
            self.L[i:,i] = np.random.normal(size=self.L.shape[0]-i)
            self.L[i,i] = abs(self.L[i,i])+np.random.rand()*3

        self.Linv = np.linalg.inv(self.L)
        self.cov = np.dot(self.L,self.L.T)
        self.covInv = np.dot(self.Linv.T,self.Linv)

    def test_choleskyDecomp(self):

        chol = linalg.jitchol(self.cov)
        self.assertTrue(np.allclose(self.L,chol))

    def test_covInv(self):

        for i in range(10):
            self.setUp()

            cinv = linalg.invert_K(self.cov)
            self.assertTrue(np.allclose(self.covInv,cinv))
