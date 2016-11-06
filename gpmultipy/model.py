import numpy as np
from kernel import RBF, White
import linalg, mvn
import scipy.stats
from freeze import Freezeable

class Model(Freezeable):

    def __init__(self,x,y,designMatrix=None):

        self.x = x
        self.y = y
        self.designMatrix = designMatrix

        self.n = self.x.shape[0]
        assert self.y.shape[0] == self.n

        if self.x.ndim == 1:
            self.x = self.x[:,None]
        self.p = self.x.shape[1]

        self.r = self.y.shape[1]

        if self.designMatrix is None:
            self.designMatrix = np.ones((1,self.r))

        assert self.designMatrix.shape[1] == self.r

        self.f = self.designMatrix.shape[0]

        self.beta = np.zeros((self.n,self.f))

        Freezeable.__init__(self,'beta')

    def function(self,f=0):
        return self.beta[:,f]

    def dataLikelihood(self,yKernel,*args,**kwargs):
        cov = yKernel.K(self.x,*args,**kwargs)
        mu = np.dot(self.beta,self.designMatrix)

        ll = 1
        for i in range(self.r):
            ll += scipy.stats.multivariate_normal.logpdf(self.y[:,i],mu[:,i],cov)
            # ll += mvn.logpdf(self.y[:,i],mu[:,i],cov)

        return ll

    def residual(self,f=None):

        r = self.y - np.dot(self.beta,self.designMatrix)

        if not f is None:
            r += np.dot(self.beta[:,f][:,None],self.designMatrix[f,:][None,:])
            r = r/self.designMatrix[f,:]
            r = r[:,self.designMatrix[f,:]!=0]

        return r
