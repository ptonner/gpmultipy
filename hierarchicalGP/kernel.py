import numpy as np
import linalg
from freeze import Freezeable

class Kernel(object):

    def __init__(self,p):
        self.p = p

    def K_inv(self,X,*args,**kwargs):

        K = self.K(X,*args,**kwargs)

        try:
            chol = linalg.jitchol(K)
            chol_inv = np.linalg.inv(chol)
        except np.linalg.linalg.LinAlgError,e:
            logger = logging.getLogger(__name__)
            logger.error('Kernel inversion error: %s'%str(self.parameters))
            raise(e)
        inv = np.dot(chol_inv.T,chol_inv)

        return inv

class RBF(Kernel,Freezeable):

    @staticmethod
    def dist(X,lengthscale):
        X = X/lengthscale

        Xsq = np.sum(np.square(X),1)
        r2 = -2.*np.dot(X, X.T) + Xsq[:,None] + Xsq[None,:]
        r2 = np.clip(r2, 0, np.inf)
        return np.sqrt(r2)

    def __init__(self,p,sigma=1,lengthscale=1):
        Kernel.__init__(self,p)
        Freezeable.__init__(self,*['sigma','lengthscale'])

        self.sigma = sigma
        self.lengthscale = lengthscale

    def K(self,X,sigma=None,lengthscale=None):

        if sigma is None:
            sigma = self.sigma
        if lengthscale is None:
            lengthscale = self.lengthscale

        dist = RBF.dist(X,lengthscale)
        return sigma*np.exp(-.5*dist**2)

class White(Kernel,Freezeable):

    def __init__(self,p,sigma=1):
        Kernel.__init__(self,p)
        Freezeable.__init__(self,*['sigma'])
        self.sigma = sigma

    def K(self,X,sigma=None):
        if sigma is None:
            sigma = self.sigma

        return sigma*np.eye(X.shape[0])
