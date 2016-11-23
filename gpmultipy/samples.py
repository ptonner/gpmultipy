import numpy as np

class Samples(object):

    @staticmethod
    def extract(samples,names):
        if len(names) == 0:
            return samples
        return Samples.extract([s[names[0]] for s in samples],names[1:])

    def __init__(self,samples,*args):
        self.samples = Samples.extract(samples,args)
        self.n = len(samples)

    def __len__(self,):
        return len(self.samples)

class LambdaSamples(Samples):
    def __init__(self,func,samples,*args):
        Samples.__init__(self,samples,*args)
        self.samples = [func(s) for s in self.samples]

class ArraySamples(LambdaSamples):
    def __init__(self,f,samples,*args):
        LambdaSamples.__init__(self,lambda x: x[:,f],samples,*args)
        self.array = np.array(self.samples)
