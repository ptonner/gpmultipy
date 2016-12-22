import numpy as np

class Parameter(object):

    def __init__(self,obj,name):
        self.obj = obj
        self.name = name

    def get(self):
        return self.__call__()

    def __call__(self):
        return self.obj.__dict__[self.name]

    def set(self,value):
        self.obj.__dict__[self.name] = value

    def __repr__(self):
        return "%s.%s"%(self.obj.__repr__(),self.name)

class ArrayParameter(Parameter):

    def __init__(self,obj,name,index=None):
        Parameter.__init__(self,obj,name)

        assert type(Parameter.__call__(self) == np.ndarray), 'must pass np.ndarray'

        self.index = index
        if self.index is None:
            self.index = range(Parameter.__call__(self).shape[1])
            # print self.index

    def __call__(self,):
        arr = Parameter.__call__(self)
        return arr[:,self.index]

    def set(self,value):
        self.obj.__dict__[self.name][:,self.index] = value
