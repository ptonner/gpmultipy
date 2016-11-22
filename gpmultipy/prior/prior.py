from ..kernel import RBF, White
from ..sampler.sampler import Sampler
from .. import linalg, mvn
import numpy as np
import scipy.stats

class Prior(Sampler):

    def __init__(self,x,kernel,functions=[],mu=None,smoothCovariance=False):
        self.x=x
        self.n = self.x.shape[0]

        self.kernel = kernel
        self.functions = functions

        self.mu = mu

        if self.mu is None:
            self.mu = np.zeros(self.n)[:,None]

        self.smoothCovariance = smoothCovariance

    def loglikelihood(self,obs,*args,**kwargs):

        assert obs.shape[0] == self.n

        if obs.shape[1] != len(self.functions):
            obs = obs[:,self.functions]

        cov = self.kernel.K(self.x,*args,**kwargs)

        #solution 1
        # cov += cov.mean()*np.eye(self.n)*1e-6

        # solution 2
        chol = linalg.jitchol(cov)
        cov = np.dot(chol,chol.T)

        rv = scipy.stats.multivariate_normal(self.mu,cov)

        ll = 1
        for i in range(obs.shape[1]):

            try:
                ll += rv.logpdf(obs[:,i])
            except:
                return -np.inf

        return ll

    def functionParameters(self,m,yKernel,f=0):

        resid = m.residual(f)
        n = np.power(m.designMatrix[f,:],2)
        n = n[n!=0]
        missingValues = np.isnan(resid)

        n = n[None,:]
        resid = np.nansum(resid*n,1)

        n = np.sum(n)
        # n = np.sum(((~missingValues)*n).T,0)

        y_inv = yKernel.K_inv(self.x)
        f_inv = self.kernel.K_inv(self.x)

        A = n*y_inv + f_inv
        b = np.dot(y_inv,resid)

        chol_A = linalg.jitchol(A)
        chol_A_inv = np.linalg.inv(chol_A)
        A_inv = np.dot(chol_A_inv.T,chol_A_inv)

        mu,cov = np.dot(A_inv,b), A_inv

        if self.smoothCovariance:
            chol = linalg.jitchol(cov)
            cov = np.dot(chol,chol.T)

        return mu,cov

    def _sample(self,m,yKernel):

        for f in self.functions:
            mu,cov = self.functionParameters(m,yKernel,f)
            m.beta[:,f] = scipy.stats.multivariate_normal.rvs(mu,cov)
            # m.beta[:,f] = scipy.stats.multivariate_normal.rvs(mu,cov)
