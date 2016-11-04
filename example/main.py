import pathhack
from gpmultipy.model import Model
from gpmultipy.prior import Prior
from gpmultipy.kernel import RBF, White
from gpmultipy.freeze import Freezer
from gpmultipy.sampler.slice import Slice
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1,1)[:,None]

yKernel = White(1,.05)
k1 = RBF(1,1,.5)

p = 20
# dm = np.zeros((2,p))
# dm[0,:p/2] = 1
# dm[1,p/2:] = 1
dm = np.ones((1,p))

prior = Prior(x,k1,range(dm.shape[0]))

fsample = scipy.stats.multivariate_normal.rvs(prior.mu,k1.K(x),size=dm.shape[0]).T
if fsample.ndim == 1:
    fsample = fsample[:,None]

mu = np.dot(fsample,dm)
y = np.array([scipy.stats.multivariate_normal.rvs(mu[:,i],yKernel.K(x)).T for i in range(p)]).T

model = Model(x,y,dm)

ySigmaSlice = Slice('ySigma',
                    lambda x: model.dataLikelihood(yKernel,sigma=x),
                    lambda x: scipy.stats.uniform(1e-6,1e1).logpdf(x),
                    .1,2,logspace=True)

kSigmaSlice = Slice('kSigma',
                    lambda x: prior.loglikelihood(model.beta,sigma=x),
                    lambda x: scipy.stats.uniform(1e-2,1e1).logpdf(x),
                    .2,5,logspace=True)

kLengthscaleSlice = Slice('kLengthscale',
                    lambda x: prior.loglikelihood(model.beta,lengthscale=x),
                    lambda x: scipy.stats.uniform(1e-2,1e1).logpdf(x),
                    .2,5,logspace=True)

samples = []
freeze = Freezer(yKernel=yKernel,k1=k1,model=model)

thin = 10
burnin = 0
nsample = 1000

ll = []

for i in range(nsample):
    prior.sample(model,yKernel)

    yKernel.sigma = ySigmaSlice.sample(yKernel.sigma)
    k1.sigma = kSigmaSlice.sample(k1.sigma)
    k1.lengthscale = kLengthscaleSlice.sample(k1.lengthscale)

    if i % thin == 0 and i > burnin:
        print model.dataLikelihood(yKernel), k1.sigma, k1.lengthscale
        samples.append(freeze.freeze())
        ll.append(model.dataLikelihood(yKernel))


# mu,cov = prior.functionParameters(model,yKernel,0)
# plt.imshow(cov,interpolation='none')

# plt.plot(np.array([s['model']['beta'][:,0] for s in samples]).T,c='r',alpha=.9)
# plt.plot(mu,c='k',lw=3)
# plt.plot(fsample[:,0])
# plt.plot(model.residual(0))

plt.subplot(231)
plt.plot(y)

plt.subplot(232)
plt.plot(np.array([s['model']['beta'][:,0] for s in samples]).T,c='r',alpha=.5)
plt.plot(fsample[:,0])

plt.subplot(233)
plt.plot(ll)
# plt.plot(np.array([s['model']['beta'][:,1] for s in samples]).T,c='r',alpha=.5)
# plt.plot(fsample[:,1])

plt.subplot(234)
# plt.plot([s['yKernel']['sigma'] for s in samples])
plt.hist([s['yKernel']['sigma'] for s in samples])

plt.subplot(235)
# plt.plot([s['k1']['sigma'] for s in samples])
plt.hist([s['k1']['sigma'] for s in samples],bins=20)
# plt.hist(np.log10([s['k1']['sigma'] for s in samples]))

plt.subplot(236)
# plt.plot([s['k1']['lengthscale'] for s in samples])
plt.hist([s['k1']['lengthscale'] for s in samples])
# plt.hist(np.log10([s['k1']['lengthscale'] for s in samples]))

plt.show()
