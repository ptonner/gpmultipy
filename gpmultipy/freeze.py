import types, copy, json
import numpy as np

def printSample(sample,depth=1):
    keys = sample.keys()
    keys.sort()

    for k in keys:
        if type(sample[k]) == dict:
            print "\t".join([""]*depth) + k
            printSample(sample[k],depth=depth+1)
        else:
            print "\t".join([""]*depth) + "%s: %s"%(k,str(sample[k]))

class Freezeable(object):

    def __init__(self,*args,**kwargs):
        self.targets = args

    def freeze(self,):
        ret = {}

        for a in self.targets:
            if issubclass(type(self.__dict__[a]), Freezeable):
                ret[a] = self.__dict__[a].freeze()
            elif type(self.__dict__[a])==types.FunctionType:
                ret[a] = self.__dict__[a]()
            else:
                ret[a] = copy.copy(self.__dict__[a])

        return ret

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

    def save(self,samples,fname):
        samples = [copy.deepcopy(s) for s in samples]

        for s in samples:
            for k,o in self.objects.iteritems():
                if k in s:
                    for k2,v in o.freeze().iteritems():
                        if type(v)==np.ndarray:
                            s[k][k2] = s[k][k2].tolist()

        s = json.dumps(samples)
        ofile = open(fname,'w')
        ofile.write(s)
        ofile.close()
