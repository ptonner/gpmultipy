from hierarchicalGP.model import Model
from hierarchicalGP.prior import Prior
from hierarchicalGP.kernel import RBF, White
from hierarchicalGP.freeze import Freezer
from hierarchicalGP.sampler.slice import Slice
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1,1)[:,None]

yKernel = White(1,.1)
k1 = RBF(1)
prior = Prior(x,k1,[0])

fsample = scipy.stats.multivariate_normal.rvs(prior.mu,k1.K(x))
y = scipy.stats.multivariate_normal.rvs(fsample,yKernel.K(x),size=10).T

model = Model(x,y,)

ySigmaSlice = Slice('ySigma',
                    lambda x: model.dataLikelihood(yKernel,sigma=x),
                    lambda x: scipy.stats.uniform(1e-6,1e1).logpdf(x),
                    1,2,logspace=True)

kSigmaSlice = Slice('kSigma',
                    lambda x: prior.loglikelihood(model.beta),
                    lambda x: scipy.stats.uniform(1e-2,1e1).logpdf(x),
                    1,2,logspace=True)

samples = []
freeze = Freezer(yKernel=yKernel,k1=k1,model=model)

for i in range(10):

    # mu,cov = prior.functionParameters(model,yKernel,0)
    # sample[0] = scipy.stats.multivariate_normal.rvs(mu,cov)
    prior.sample(model,yKernel)

    yKernel.sigma = ySigmaSlice.sample(yKernel.sigma)
    k1.sigma = kSigmaSlice.sample(k1.sigma)

    print model.dataLikelihood(yKernel)
    samples.append(freeze.freeze())


plt.plot(np.array([s['model'][0][:,0] for s in samples]).T)
plt.show()
