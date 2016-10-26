import pathhack
from gpmultipy.model import Model
from gpmultipy.prior import Prior
from gpmultipy.kernel import RBF, White, Addition
from gpmultipy.freeze import Freezer
from gpmultipy.sampler.slice import Slice
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1,1)[:,None]

# yKernel = RBF(1,sigma=.1,lengthscale=.3)
theta = .7
sigma = .1
yKernel = Addition(White(1,sigma=(1-theta)*sigma),RBF(1,sigma=theta*sigma,lengthscale=.3))

k1 = RBF(1,sigma=.5,lengthscale=.5)
k2 = RBF(1,sigma=.25,lengthscale=1)

p = 3
b = 5
dm = np.zeros((1+b,p*b))
dm[0,:] = 1
for i in range(b):
    dm[i+1,i*p:(i+1)*p] = 1

prior = Prior(x,k1,[0])
prior2 = Prior(x,k2,range(1,dm.shape[0]))

fsample = np.zeros((50,dm.shape[0]))
fsample[:,0] = scipy.stats.multivariate_normal.rvs(prior.mu,k1.K(x))
for i in range(b):
    fsample[:,i+1] = scipy.stats.multivariate_normal.rvs(prior2.mu,k2.K(x))

mu = np.dot(fsample,dm)
y = np.array([scipy.stats.multivariate_normal.rvs(mu[:,i],yKernel.K(x)).T for i in range(mu.shape[1])]).T

model = Model(x,y,dm)

ySigmaSlice = Slice('ySigma',
                    lambda x: model.dataLikelihood(yKernel,k1_sigma=x),
                    lambda x: scipy.stats.uniform(1e-6,1e1).logpdf(x),
                    1,2,logspace=True)

ySigma2Slice = Slice('ySigma2',
                    lambda x: model.dataLikelihood(yKernel,k2_sigma=x),
                    lambda x: scipy.stats.uniform(1e-6,1e1).logpdf(x),
                    1,2,logspace=True)

yLengthscaleSlice = Slice('yLengthscale',
                    lambda x: model.dataLikelihood(yKernel,k2_lengthscale=x),
                    lambda x: scipy.stats.uniform(1e-6,1e1).logpdf(x),
                    1,2,logspace=True)

kSigmaSlice = Slice('kSigma',
                    lambda x: prior.loglikelihood(model.beta,sigma=x),
                    lambda x: scipy.stats.uniform(1e-2,1e1).logpdf(x),
                    .1,10,logspace=True)

kLengthscaleSlice = Slice('kLengthscale',
                    lambda x: prior.loglikelihood(model.beta,lengthscale=x),
                    lambda x: scipy.stats.uniform(1e-2,1e1).logpdf(x),
                    .1,10,logspace=True)

k2SigmaSlice = Slice('k2Sigma',
                    lambda x: prior2.loglikelihood(model.beta,sigma=x),
                    lambda x: scipy.stats.uniform(1e-2,1e1).logpdf(x),
                    .1,10,logspace=True)

k2LengthscaleSlice = Slice('k2Lengthscale',
                    lambda x: prior2.loglikelihood(model.beta,lengthscale=x),
                    lambda x: scipy.stats.uniform(1e-2,1e1).logpdf(x),
                    .1,10,logspace=True)

samples = []
freeze = Freezer(yK1=yKernel.k1,yK2=yKernel.k2,k1=k1,model=model)

thin = 1
burnin = 0
nsample = 10

for i in range(nsample):
    prior.sample(model,yKernel)
    prior2.sample(model,yKernel)

    yKernel.k1.sigma = ySigmaSlice.sample(yKernel.k1.sigma)
    yKernel.k2.sigma = ySigma2Slice.sample(yKernel.k2.sigma)
    yKernel.k2.lengthscale = yLengthscaleSlice.sample(yKernel.k2.lengthscale)

    k1.sigma = kSigmaSlice.sample(k1.sigma)
    k1.lengthscale = kLengthscaleSlice.sample(k1.lengthscale)

    k2.sigma = k2SigmaSlice.sample(k2.sigma)
    k2.lengthscale = k2LengthscaleSlice.sample(k2.lengthscale)

    if i % thin == 0 and i > burnin:
        print model.dataLikelihood(yKernel), k1.sigma, k1.lengthscale
        samples.append(freeze.freeze())


nrow = 2
ncol = max(b+2,3)

plt.subplot(nrow,ncol,1)
plt.plot(y)

plt.subplot(nrow,ncol,2)
plt.plot(np.array([s['model']['beta'][:,0] for s in samples]).T,c='r',alpha=.5)
plt.plot(fsample[:,0])

for i in range(b):
    plt.subplot(nrow,ncol,3+i)
    plt.plot(np.array([s['model']['beta'][:,i+1] for s in samples]).T,c='r',alpha=.5)
    plt.plot(fsample[:,i+1])

# plt.subplot(233)
# plt.plot(np.array([s['model']['beta'][:,1] for s in samples]).T,c='r',alpha=.5)
# plt.plot(fsample[:,1])

plt.subplot(nrow,ncol,ncol+1)
# plt.plot([s['yKernel']['sigma'] for s in samples])
# plt.hist([s['yKernel']['sigma'] for s in samples])
plt.hist([s['yK1']['sigma'] for s in samples])

plt.subplot(nrow,ncol,ncol+2)
# plt.plot([s['k1']['sigma'] for s in samples])
# plt.hist([s['k1']['sigma'] for s in samples])
plt.hist(np.log([s['k1']['sigma'] for s in samples]))

plt.subplot(nrow,ncol,ncol+3)
# plt.plot([s['k1']['lengthscale'] for s in samples])
# plt.hist([s['k1']['lengthscale'] for s in samples])
plt.hist(np.log([s['k1']['lengthscale'] for s in samples]))

plt.show()
