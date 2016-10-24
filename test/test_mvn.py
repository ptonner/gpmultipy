import unittest
from hierarchicalGP import linalg,mvn
import numpy as np
import scipy

class TestMVN(unittest.TestCase):

    def setUp(self,):
        self.L = np.zeros((50,50))

        for i in range(self.L.shape[0]):
            self.L[i:,i] = np.random.normal(size=self.L.shape[0]-i)
            self.L[i,i] = abs(self.L[i,i])+np.random.rand()*3

        self.cov = np.dot(self.L,self.L.T)

    def test_logpdf(self):
        mu = np.zeros(50)
        sample = np.random.normal(size=50)
        pdf1 = mvn.logpdf(sample,mu,L=self.L)
        pdf2 = scipy.stats.multivariate_normal.logpdf(sample,mu,self.cov)

        self.assertTrue(np.allclose(pdf1,pdf2))
