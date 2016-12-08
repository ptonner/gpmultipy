# cython: profile=True

from . import Sampler
import numpy as np
import scipy

class Slice(Sampler):
    """Slice sampling as described in Neal (2003), using the 'step-out' algorithm."""

    def __init__(self,name,logdensity_fxn,prior_logdensity_fxn=None,w=1,m=3,logspace=False):
        """Args:
        	logdensity_fxn: logarithm of conditional density function of the parameter X that we will evaluate to find our slice interval
        	w: an interval step size, defining how large each interval increase is
        	m: limits the maximum interval size to w*m

        Returns:
        	x1: the new sample of the variable X
        	l,r: the final region bounds used
        """
        Sampler.__init__(self,name,'Slice')
        self.logdensity_fxn = logdensity_fxn

        self.prior_logdensity_fxn = prior_logdensity_fxn
        if self.prior_logdensity_fxn is None:
            self.prior_logdensity_fxn = lambda x: 0

        self.w = w
        self.m = m
        self.logspace = logspace

    def loglikelihood(self,x,useLogspace=True):
        if useLogspace and self.logspace:
            x = pow(10,x)

        try:
            pldf = self.prior_logdensity_fxn(x)
            # if pldf == -np.inf:
            #     print 'fail'
            #     return -np.inf

            return self.logdensity_fxn(x) + pldf
        except:
            return -np.inf

    def _sample(self, x):

        if self.logspace:
            x = np.log10(x)

        # cdef double z, u, v, l, r, f0, x1;
        # cdef int j,k;

        f0 = self.loglikelihood(x)
        z = f0 - scipy.stats.expon.rvs(1)

        # find our interval
        u = scipy.stats.uniform.rvs(0,1)
        l = x-self.w*u
        r = l+self.w

        v = scipy.stats.uniform.rvs(0,1)
        j = int(self.m*v)
        k = self.m-1-j

        while j > 0 and z < self.loglikelihood(l):
            j -= 1
            l -= self.w

        while k > 0 and z < self.loglikelihood(r):
            k -= 1
            r += self.w


        # pick a new point
        u = scipy.stats.uniform.rvs(0,1)
        x1 = l + u*(r-l)

        while z > self.loglikelihood(x1):
            if x1 < x:
                l = x1
            else:
                r = x1

            u = scipy.stats.uniform.rvs(0,1)
            x1 = l + u*(r-l)

        if self.logspace:
            x1 = pow(10,x1)

        return x1


class PriorSlice(Slice):

    def __init__(self,name,model,functionPrior,paramPrior,param,*args,**kwargs):
        self.model = model
        self.functionPrior = functionPrior
        self.paramPrior = paramPrior
        self.param = param

        Slice.__init__(self,name,self.observation_loglikelihood,self.prior_loglikelihood,*args,**kwargs)

    def observation_loglikelihood(self,x):
        return self.functionPrior.loglikelihood(self.model.beta,**{self.param:x})

    def prior_loglikelihood(self,x):
        return self.paramPrior.logpdf(x)
