from kernel import Kernel
from ..freeze import Freezeable

class Addition(Kernel,Freezeable):

    def __init__(self,k1,k2,*args,**kwargs):
        self.k1 = k1
        self.k2 = k2
        # params = []
        # for k in [self.k1,self.k2]:
        #     params.extend([v for k,v in k.parameters.items()])

        assert self.k1.p == self.k2.p

        Kernel.__init__(self,self.k1.p,*args,**kwargs)
        Freezeable.__init__(self,'k1','k2')

    def K(self,X,**kwargs):

        kwargs1 = {}
        kwargs2 = {}

        for k,v in kwargs.iteritems():
            if k[:3] == 'k1_':
                kwargs1[k[3:]] = v
            elif k[:3] == 'k2_':
                kwargs2[k[3:]] = v

        k1 = self.k1.K(X,**kwargs1)
        k2 = self.k2.K(X,**kwargs2)

        return k1 + k2
