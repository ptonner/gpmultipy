from bayesGP.model import Model
from bayesGP.prior import Prior
from bayesGP.kernel import RBF, White
import scipy.stats
import numpy as np

x = np.linspace(-1,1)[:,None]

yKernel = White(1,.1)
k1 = RBF(1)
prior = Prior(x,k1,[0])

fsample = scipy.stats.multivariate_normal.rvs(prior.mu,k1.K(x))
y = scipy.stats.multivariate_normal.rvs(fsample,yKernel.K(x),size=10).T

model = Model(x,y,)

samples = []

for i in range(100):
    sample = {}

    mu,cov = prior.functionParameters(model,yKernel,0)
    sample[0] = scipy.stats.multivariate_normal.rvs(mu,cov)

    

# import matplotlib.pyplot as plt
# plt.plot(np.array(samples).T)
# plt.show()
