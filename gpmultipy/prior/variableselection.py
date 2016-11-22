from prior import Prior
import numpy as np
import scipy

class VariableSelection(Prior):

    def __init__(self,x,kernel,functions,mu=None,theta=.5,default=True,**kwargs):
        """Variable selection prior on functions.

        theta: prior (Bernoulli) probability of inclusion
        default: default inclusion or exclusion of functions"""

        Prior.__init__(self,x,kernel,functions,mu,**kwargs)
        self.theta = theta
        self.toggle = [default]*len(self.functions)
        self.priorrv = scipy.stats.bernoulli(self.theta)

    def marginalLikelihood(self,model,yKernel,f,included=True):
        """The marginal likelihood of data depending on function f."""

        select = model.designMatrix[f,:] != 0
        ind = np.where(select)[0]
        num = select.sum()

        if included:
            mu = np.repeat(self.mu,select.sum(),1).ravel(1)
        else:
            mu = np.repeat(np.zeros(self.n)[:,None],select.sum(),1).ravel(1)

        cov = np.zeros((self.n*select.sum(),self.n*select.sum()))

        for i in range(select.sum()):
            for j in range(select.sum()):
                if i == j:
                    cov[i*self.n:(i+1)*self.n,j*self.n:(j+1)*self.n] += yKernel.K(self.x)

                if included:
                    cov[i*self.n:(i+1)*self.n,j*self.n:(j+1)*self.n] += self.kernel.K(self.x) * model.designMatrix[f,ind[i]] * model.designMatrix[f,ind[j]]

        # data = np.zeros(self.n*num)
        # for i in range(num):
        #     data[i*self.n:(i+1)*self.n] = model.y[:,ind[i]] / model.designMatrix[f,ind[i]]
        data = model.residual(f).ravel(1)

        rv = scipy.stats.multivariate_normal(mu,cov)
        return rv.logpdf(data)

    def inclusionLikelihood(self,model,yKernel,f=None):
        """The probability of including a function.

        Computed as p(Y|f!=0)*p(f!=0) / (p(Y|f!=0)*p(f!=0) + p(Y|f=0)*p(f=0))"""

        if f is None:
            f = self.functions[0]

        p1 = self.marginalLikelihood(model,yKernel,f)
        p1 += self.priorrv.logpmf(1)

        p2 = self.marginalLikelihood(model,yKernel,f,included=False)
        p2 += self.priorrv.logpmf(0)

        # p1 = np.exp(p1)
        # p2 = np.exp(p2)

        # if np.isnan(p1):
        #     p1 = 0
        # if np.isnan(p2):
        #     p2 = 0
        # if p1 == 0 and p2 == 0:
        #     raise ValueError("Both probabilities are 0!")

        # p = p1 / (p1 + p2)
        logp = p1 - np.logaddexp(p1,p2)
        p = np.exp(logp)

        return p
