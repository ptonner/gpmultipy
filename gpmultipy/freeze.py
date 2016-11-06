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
        if name in self.targets:
            self.__dict__[name] = value
        else:
            raise AttributeError("attribute %s is not in targets!!"%name)


class Freezer(object):
    """brrrrrrrrrrr."""

    def __init__(self,*args,**kwargs):
        self.objects = kwargs

    def evalFunctionWithSample(self,sample,f):

        current = self.freeze()
        self.push(**sample)
        ret = f()
        self.push(**current)
        return ret

    def push(self,**kwargs):

        for k,v in kwargs.iteritems():
            if k in self.objects:
                for kk,vv in v.iteritems():
                    self.objects[k].update(kk,vv)

    def freeze(self,):

        ret = {}
        for k,v in self.objects.iteritems():
            ret[k] = v.freeze()

        return ret
