from kernel import Kernel
from ..freeze import Freezeable
import numpy as np

class White(Kernel,Freezeable):

    def __init__(self,p,sigma=1):
        Kernel.__init__(self,p)
        Freezeable.__init__(self,'sigma')
        self.sigma = sigma

    def K(self,X,sigma=None):
        if sigma is None:
            sigma = self.sigma

        return sigma*np.eye(X.shape[0])
