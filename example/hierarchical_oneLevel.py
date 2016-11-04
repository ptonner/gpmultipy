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

yKernel = White(1,sigma=.005)

theta = .9
sigma = .5
k1 = RBF(1,sigma=sigma*theta,lengthscale=.5)
k2 = RBF(1,sigma=sigma*(1-theta),lengthscale=1)

p = 2
b = 10
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
                    lambda x: model.dataLikelihood(yKernel,sigma=x),
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
freeze = Freezer(yKernel=yKernel,k1=k1,k2=k2,model=model)

thin = 10
burnin = 0
nsample = 4000
ll = []
for i in range(nsample):
    prior.sample(model,yKernel)
    prior2.sample(model,yKernel)

    yKernel.sigma = ySigmaSlice.sample(yKernel.sigma)

    k1.sigma = kSigmaSlice.sample(k1.sigma)
    k1.lengthscale = kLengthscaleSlice.sample(k1.lengthscale)

    k2.sigma = k2SigmaSlice.sample(k2.sigma)
    k2.lengthscale = k2LengthscaleSlice.sample(k2.lengthscale)

    if i % thin == 0 and i > burnin:
        samples.append(freeze.freeze())
        print model.dataLikelihood(yKernel), k1.sigma, k1.lengthscale, k2.sigma, k2.lengthscale
        # print samples[-1]

        ll.append(model.dataLikelihood(yKernel))


nrow = 4
ncol = max(b+1,5)

# plt.subplot(nrow,ncol,1)
plt.subplot2grid((nrow,ncol),(0,0),colspan=ncol,rowspan=2)
plt.plot(y)
plt.plot(fsample[:,0])

# plt.subplot(nrow,ncol,2)
plt.subplot2grid((nrow,ncol),(2,0))
plt.plot(np.array([s['model']['beta'][:,0] for s in samples]).T,c='r',alpha=.5)
plt.plot(fsample[:,0])

for i in range(b):
    # plt.subplot(nrow,ncol,3+i)
    plt.subplot2grid((nrow,ncol),(2,i+1))
    plt.plot(np.array([s['model']['beta'][:,i+1] for s in samples]).T,c='r',alpha=.5)
    plt.plot(fsample[:,i+1])

    plt.ylim(fsample.min()-.2,fsample.max()+.2)

plt.subplot2grid((nrow,ncol),(3,0),colspan=2)
plt.hist([s['yKernel']['sigma'] for s in samples])

plt.subplot2grid((nrow,ncol),(3,2),colspan=2)
# plt.hist(np.log10([s['k1']['sigma'] for s in samples]))
plt.hist([s['k1']['sigma'] for s in samples])

plt.subplot2grid((nrow,ncol),(3,4),colspan=2)
# plt.hist(np.log10([s['k1']['lengthscale'] for s in samples]))
plt.hist([s['k1']['lengthscale'] for s in samples])

plt.subplot2grid((nrow,ncol),(3,6),colspan=2)
# plt.hist(np.log10([s['k2']['sigma'] for s in samples]))
plt.hist([s['k2']['sigma'] for s in samples])

plt.subplot2grid((nrow,ncol),(3,8),colspan=2)
# plt.hist(np.log10([s['k2']['lengthscale'] for s in samples]))
plt.hist([s['k2']['lengthscale'] for s in samples])

plt.subplot2grid((nrow,ncol),(3,9),)
plt.plot(ll)


plt.show()
