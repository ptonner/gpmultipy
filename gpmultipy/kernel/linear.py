from kernel import Kernel
from ..freeze import Freezeable
import numpy as np

class Linear(Kernel,Freezeable):

    def K(self,X,sigma=None,lengthscale=None):

        if sigma is None:
            sigma = self.sigma
        if lengthscale is None:
            lengthscale = self.lengthscale

        X = X/lengthscale
        out = np.dot(X,X.T)
        return sigma*out

    def __init__(self,p,sigma=1,lengthscale=1):

        Kernel.__init__(self,p,)
        Freezeable.__init__(self,'sigma','lengthscale')
        self.sigma = sigma
        self.lengthscale = lengthscale
