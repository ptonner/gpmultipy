import unittest
from gpmultipy import Freezer, Freezeable
import numpy as np

class TestFreezeable(Freezeable):

    def __init__(self,*args):
        targets = []
        for i,a in enumerate(args):
            self.__dict__['param{}'.format(i)] = a
            targets.append('param{}'.format(i))

        Freezeable.__init__(self,*targets)
        self.p = len(args)

    def checkFreeze(self,f):

        for i in range(self.p):
            name = 'param{}'.format(i)
            if not name in f:
                return False
            if not f[name] == self.__dict__[name]:
                return False
            # if not self.__dict__[name] == f[i]:
            #     return False

        return True

class TestFreezer(unittest.TestCase):

    def __init__(self,*args,**kwargs):
        unittest.TestCase.__init__(self,*args,**kwargs)

    def setUp(self,):

        self.freezeable = TestFreezeable(1)
        self.freezer = Freezer(freezeable=self.freezeable)

    def test_freeze(self):

        freeze = self.freezer.freeze()

        self.assertIn('freezeable',freeze)
        self.assertTrue(self.freezeable.checkFreeze(freeze['freezeable']))
