from kernel import Kernel
from ..freeze import Freezeable
import numpy as np

class RBF(Kernel,Freezeable):

    @staticmethod
    def dist(X,lengthscale):
        X = X/lengthscale

        Xsq = np.sum(np.square(X),1)
        r2 = -2.*np.dot(X, X.T) + (Xsq[:,None] + Xsq[None,:])
        r2 = np.clip(r2, 0, np.inf)
        return np.sqrt(r2)

    def __init__(self,p,sigma=1,lengthscale=1):
        Kernel.__init__(self,p)
        Freezeable.__init__(self,'sigma','lengthscale')

        self.sigma = sigma
        self.lengthscale = lengthscale

    def K(self,X,sigma=None,lengthscale=None):

        if sigma is None:
            sigma = self.sigma
        if lengthscale is None:
            lengthscale = self.lengthscale

        dist = RBF.dist(X,lengthscale)
        return sigma*np.exp(-.5*dist**2)
