from interval import Interval
import numpy as np
from copy import deepcopy

class ScalarInterval(Interval):

    def __init__(self,name,parent,samples,alpha,pi,freezer,logspace=False):
        Interval.__init__(self,samples,alpha,ndim=1)

        self.name = name
        self.parent = parent

        # method for computing p(theta | data)
        self.pi = pi

        self.freezer = freezer

        # compute the hpd interval
        # Adapted from "Monte Carlo Estimation of Bayesian Credible and HPD Intervals", Ming-Hui Chen and Qi-Man Shao, 1999
        # self.epsilon = [self.pi(t) for t in self.samples]
        self.epsilon = [self.compute_pi(sample=t) for t in self.samples]
        self.epsilon.sort()
        j = int(self.n*self.alpha)
        self.epj = self.epsilon[j] # any observation with pi(x|D) > epj is in the region

        self.paramSamples = [s[parent][name] for s in self.samples]
        self.paramSamples.sort()
        self.lb = self.paramSamples[int(self.n*self.alpha)]
        self.ub = self.paramSamples[int(self.n*(1-self.alpha))]

    def compute_pi(self,x=None,sample=None):
        if sample is None:
            sample = deepcopy(self.samples[-1])
        if not x is None:
            sample[self.parent][self.name] = x

        return self.freezer.evalFunctionWithSample(sample,lambda: self.pi(sample[self.parent][self.name]))

    def contains(self,x):
        #return self.compute_pi(x) > self.epj

        return x > self.lb and x < self.ub

    def plot(self,lims,x=None,offset=-1e-3,logspace=False):

        import matplotlib.pyplot as plt

        on = False
        regs = []
        for z in np.linspace(*lims):
            e = z
            if logspace:
                e = pow(10,e)

            if not on and self.contains(e):
                on = True
                start = z
            elif on and not self.contains(e):
                on = False
                regs.append((start,z))

        for r in regs:
            plt.hlines(offset,r[0],r[1],lw=3)

        if not x is None:
            plt.scatter(x,offset*2,c='r',marker='x',s=50)

        plt.yticks([])
        plt.ylim(-.0015,.0005)
