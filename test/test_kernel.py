import unittest
from hierarchicalGP.kernel import Kernel, RBF, White
from hierarchicalGP import linalg
import numpy as np

class TestingKernel(Kernel):
    def K(self,x):
        return np.eye(x.shape[0])

class TestKernel(unittest.TestCase):

    def __init__(self,*args,**kwargs):
        unittest.TestCase.__init__(self,*args,**kwargs)
        self.kernelType=TestingKernel
        self.kernelKwargs = {}
        self.p=1

    def setUp(self,):

        self.x = np.repeat(np.linspace(-1,1)[:,None],self.p,axis=1)
        self.kernel = self.kernelType(self.p,**self.kernelKwargs)

    def test_invertKernel(self):
        k = self.kernel.K(self.x)
        inv = linalg.invert_K(k)
        self.assertTrue(np.allclose(self.kernel.K_inv(self.x),inv))

class TestRBF(TestKernel):
    def __init__(self,*args,**kwargs):
        TestKernel.__init__(self,*args,**kwargs)
        self.kernelType = RBF

    def test_updateLengthscale(self,):
        k1 = self.kernel.K(self.x)
        self.kernel.lengthscale = 10
        k2 = self.kernel.K(self.x)
        self.assertTrue(np.less_equal(k1,k2).all())

        self.kernel.lengthscale = .1
        k2 = self.kernel.K(self.x)
        self.assertTrue(np.less_equal(k2,k1).all())

    def test_updateSigma(self,):
        k1 = self.kernel.K(self.x)
        self.kernel.sigma = 10
        k2 = self.kernel.K(self.x)
        self.assertTrue(np.less(k1,k2).all())

        self.kernel.sigma = .1
        k2 = self.kernel.K(self.x)
        self.assertTrue(np.less(k2,k1).all())

class TestRBF_update(TestRBF):

    def __init__(self,*args,**kwargs):
        TestRBF.__init__(self,*args,**kwargs)


if __name__ == '__main__':
    unittest.main()
