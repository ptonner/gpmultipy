import unittest
from hierarchicalGP.kernel import Kernel, RBF, White, Addition, Product
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
        if self.kernel is None:
            return

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

class TestRBF(TestKernel):
    def __init__(self,*args,**kwargs):
        TestKernel.__init__(self,*args,**kwargs)
        self.kernelType = White

class TestCombination(TestKernel):
    def __init__(self,*args,**kwargs):
        TestKernel.__init__(self,*args,**kwargs)
        self.kernelType = RBF
        self.combinationKernel = None

    def setUp(self,):

        self.x = np.repeat(np.linspace(-1,1)[:,None],self.p,axis=1)
        self.k1 = self.kernelType(self.p,**self.kernelKwargs)
        self.k2 = self.kernelType(self.p,**self.kernelKwargs)

        if not self.combinationKernel is None:
            self.kernel = self.combinationKernel(self.k1,self.k2)
        else:
            self.kernel = None

class TestProduct(TestCombination):

    def __init__(self,*args,**kwargs):
        TestCombination.__init__(self,*args,**kwargs)
        self.combinationKernel = Product

    def test_kernelFunction(self,):
        k = self.k1.K(self.x)
        k *= self.k2.K(self.x)
        self.assertTrue(np.allclose(k,self.kernel.K(self.x)))

class TestAddition(TestCombination):
    def __init__(self,*args,**kwargs):
        TestCombination.__init__(self,*args,**kwargs)
        self.combinationKernel = Addition

    def test_kernelFunction(self,):
        k = self.k1.K(self.x)
        k += self.k2.K(self.x)
        self.assertTrue(np.allclose(k,self.kernel.K(self.x)))

if __name__ == '__main__':
    unittest.main()
