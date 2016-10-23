import types
import copy

class Freezeable(object):

    def __init__(self,*args,**kwargs):
        self.targets = args

    def freeze(self,):
        # return [copy.copy(self.__dict__[a]) for a in self.targets]
        return {a:copy.copy(self.__dict__[a]) for a in self.targets}
        # return {a:copy.copy(self.__dict__[a]) if not type(self.__dict__[a])==types.Function else a:self.__dict__[a]() for a in self.targets}

    def update(self,name,value):
        self.__dict__[name] = value


class Freezer(object):
    """brrrrrrrrrrr."""

    def __init__(self,*args,**kwargs):
        self.objects = kwargs

    def freeze(self,):

        ret = {}
        for k,v in self.objects.iteritems():
            ret[k] = v.freeze()

        return ret
