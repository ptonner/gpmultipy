from kernel import RBF, White
import linalg
import numpy as np
import scipy.stats

class Prior(object):

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
        cov += cov.mean()*np.eye(self.n)*1e-6
        
        ll = 0
        for i in range(obs.shape[1]):
            ll += scipy.stats.multivariate_normal.logpdf(obs[:,i],self.mu,cov)

        return ll

    def functionParameters(self,m,yKernel,f=0):

        resid = m.residual(f)
        n = m.designMatrix[f,:]

        y_inv = yKernel.K_inv(self.x)
        f_inv = self.kernel.K_inv(self.x)

        A = n.sum()*y_inv + f_inv
        b = np.dot(y_inv,resid).sum(1)

        chol_A = linalg.jitchol(A)
        chol_A_inv = np.linalg.inv(chol_A)
        A_inv = np.dot(chol_A_inv.T,chol_A_inv)

        mu,cov = np.dot(A_inv,b), A_inv

        return mu,cov

    def sample(self,m,yKernel):

        for f in self.functions:
            mu,cov = self.functionParameters(m,yKernel)
            m.beta[:,f] = scipy.stats.multivariate_normal.rvs(mu,cov)
