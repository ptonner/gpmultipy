import unittest
from gpmultipy.parameter import Parameter, ArrayParameter
import numpy as np

class Dummy(object):

    def __init__(self,*args,**kwargs):

        for k,v in kwargs.iteritems():
            self.__dict__[k] = v

class TestParameter(unittest.TestCase):

    # def __init__(self,dKwargs={'a':0},*args,**kwargs):
    #     unittest.TestCase.__init__(self,*args,**kwargs)
    #     self.dkwargs = dKwargs

    def setUp(self,):
        self.dkwargs = {'a':0}
        self.dummy = Dummy(**self.dkwargs)
        self.param = Parameter(self.dummy,'a')

    def test_return(self):

        self.assertEqual(self.param(),self.dummy.a)

    def test_update(self):
        self.param.set(1)

        self.assertEqual(1,self.dummy.a)
        self.assertEqual(self.param(),1)
        self.assertEqual(self.param(),self.dummy.a)

class TestArrayParameter(unittest.TestCase):

    def setUp(self,):
        self.dkwargs = {'a':np.random.normal(size=50).reshape((10,5))}
        self.dummy = Dummy(**self.dkwargs)

    def test_return(self):
        self.param = ArrayParameter(self.dummy,'a',)
        self.assertTrue(np.all(self.param()==self.dummy.a))

        idx = [0]
        self.param = ArrayParameter(self.dummy,'a',index=idx)
        self.assertTrue(np.all(self.param()==self.dummy.a[:,idx]))

        idx = [0,1,2]
        self.param = ArrayParameter(self.dummy,'a',index=idx)
        self.assertTrue(np.all(self.param()==self.dummy.a[:,idx]))

        idx = [0,2,4]
        self.param = ArrayParameter(self.dummy,'a',index=idx)
        self.assertTrue(np.all(self.param()==self.dummy.a[:,idx]))

if __name__ == '__main__':
    unittest.main()
