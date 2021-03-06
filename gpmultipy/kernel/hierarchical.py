from kernel import Kernel
import numpy as np
from ..freeze import Freezeable

class Hierarchical(Kernel,Freezeable):

    def __init__(self,*args):
        self.kernels = [a for a in args if issubclass(type(a),Kernel)]
        self.levels = len(self.kernels)

        args =[]
        for i,k in enumerate(self.kernels):
            name = 'k%d'%(i+1)
            self.__dict__[name] = k
            args.append(name)

        Freezeable.__init__(self,*args)

    def K(self,x,*args,**kwargs):

        ids = x[:,x.shape[1]-self.levels:]
        x = x[:,:x.shape[1]-self.levels]

        # rebuild 2 dims
        if x.ndim == 1:
            x = x[:,None]

        k = np.zeros((x.shape[0],x.shape[0]))

        for i,kern in enumerate(self.kernels):
            kw = {}
            for k,v in kwargs.iteritems():
                if k[:3] == 'k%d_'%i:
                    kw[k[3:]] = v

            for j in np.unique(ids[:,i]):
                select = np.where(ids[:,i]==j)[0]
                xselect = x[select,:]

                k[np.ix_(select,select)] += self.kernels[i].K(xselect,**kwargs)

        return k
