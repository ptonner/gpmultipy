from kernel import RBF, White
import linalg, mvn
import numpy as np
import scipy.stats
from sampler.sampler import Sampler

class Prior(Sampler):

    def __init__(self,x,kernel,functions=[],mu=None):
        self.x=x
        self.n = self.x.shape[0]

        self.kernel = kernel
        self.functions = functions

        self.mu = mu

        if self.mu is None:
            self.mu = np.zeros(self.n)

    def loglikelihood(self,obs,*args,**kwargs):

        assert obs.shape[0] == self.n

        cov = self.kernel.K(self.x,*args,**kwargs)
        chol = linalg.jitchol(cov)

        ll = 0
        for i in range(obs.shape[1]):
            ll += mvn.logpdf(obs[:,i],self.mu,L=chol)

        return ll

    def functionParameters(self,m,yKernel,f=0):

        resid = m.residual(f)
        n = m.designMatrix[f,:]
        n = n[n!=0]

        y_inv = yKernel.K_inv(self.x)
        f_inv = self.kernel.K_inv(self.x)

        A = n.sum()*y_inv + f_inv
        b = np.dot(y_inv,resid).sum(1)

        chol_A = linalg.jitchol(A)
        chol_A_inv = np.linalg.inv(chol_A)
        A_inv = np.dot(chol_A_inv.T,chol_A_inv)

        mu,cov = np.dot(A_inv,b), A_inv

        return mu,cov

    def _sample(self,m,yKernel):

        ret = np.zeros(m.beta.shape)

        for f in self.functions:
            mu,cov = self.functionParameters(m,yKernel,f)
            m.beta[:,f] = scipy.stats.multivariate_normal.rvs(mu,cov)
            # m.beta[:,f] = scipy.stats.multivariate_normal.rvs(mu,cov)
