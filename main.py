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

p = 10
dm = np.zeros((2,p))
dm[0,:p/2] = 1
dm[1,p/2:] = 1

prior = Prior(x,k1,range(dm.shape[0]))

fsample = scipy.stats.multivariate_normal.rvs(prior.mu,k1.K(x),size=dm.shape[0]).T

mu = np.dot(fsample,dm)
y = np.array([scipy.stats.multivariate_normal.rvs(mu[:,i],yKernel.K(x)).T for i in range(p)]).T

model = Model(x,y,dm)

ySigmaSlice = Slice('ySigma',
                    lambda x: model.dataLikelihood(yKernel,sigma=x),
                    lambda x: scipy.stats.uniform(1e-6,1e1).logpdf(x),
                    1,2,logspace=True)

kSigmaSlice = Slice('kSigma',
                    lambda x: prior.loglikelihood(model.beta,sigma=x),
                    lambda x: scipy.stats.uniform(1e-2,1e0).logpdf(x),
                    w=1,m=2,logspace=True)

kLengthscaleSlice = Slice('kLengthscale',
                    lambda x: prior.loglikelihood(model.beta,lengthscale=x),
                    lambda x: scipy.stats.uniform(1e-2,1e1).logpdf(x),
                    1,2,logspace=True)

samples = []
freeze = Freezer(yKernel=yKernel,k1=k1,model=model)

thin = 10
burnin = 200
nsample = 1000

for i in range(nsample):
    # break

    # mu,cov = prior.functionParameters(model,yKernel,0)
    # sample[0] = scipy.stats.multivariate_normal.rvs(mu,cov)
    prior.sample(model,yKernel)

    yKernel.sigma = ySigmaSlice.sample(yKernel.sigma)
    k1.lengthscale = kLengthscaleSlice.sample(k1.lengthscale)
    k1.sigma = kSigmaSlice.sample(k1.sigma)

    if i % thin == 0 and i > burnin:
        print model.dataLikelihood(yKernel)
        samples.append(freeze.freeze())


plt.subplot(231)
plt.plot(y)

plt.subplot(232)
plt.plot(np.array([s['model'][0][:,0] for s in samples]).T,c='r',alpha=.5)
plt.plot(fsample[:,0])

plt.subplot(233)
plt.plot(np.array([s['model'][0][:,1] for s in samples]).T,c='r',alpha=.5)
plt.plot(fsample[:,1])

plt.subplot(234)
plt.plot([s['yKernel'][0] for s in samples])

plt.subplot(235)
plt.plot([s['k1'][0] for s in samples])

plt.subplot(236)
plt.plot([s['k1'][1] for s in samples])

plt.show()
