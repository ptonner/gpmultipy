import unittest
from gpmultipy.model import Model
import numpy as np

class TestModel(unittest.TestCase):

    def __init__(self,p=5,f=1,*args,**kwargs):
        unittest.TestCase.__init__(self,*args,**kwargs)
        self.p = p
        self.f = f

    def buildDesignMatrix(self,f,p):
        return np.ones((f,p))

    def setUp(self,):
        x = np.linspace(-1,1)[:,None]
        y = np.random.normal(size=(50,self.p))
        dm = self.buildDesignMatrix(self.f,self.p)

        self.model = Model(x,y,dm)

class TestModelFunctionUpdate(TestModel):

    def __init__(self,*args,**kwargs):
        TestModel.__init__(self,5,1,*args,**kwargs)

    def test_functionUpdate(self):
        new = np.random.rand(50)
        self.model.beta[:,0] = new
        self.assertTrue(np.allclose(new,self.model.function(0)))

class TestModelResidual(TestModel):

    def __init__(self,*args,**kwargs):
        TestModel.__init__(self,5,3,*args,**kwargs)

    def test_residual_default(self):
        self.assertTrue(np.allclose(self.model.residual(),self.model.y))

    def test_residual_equal_data(self):
        self.model.beta[:,0] = self.model.y[:,0]
        self.assertTrue(np.allclose(self.model.residual()[:,0],np.zeros(50)))
        self.assertTrue(np.allclose(self.model.residual(0)[:,0],self.model.y[:,0]))


if __name__ == '__main__':
    unittest.main()
