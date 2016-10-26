import numpy as np
import linalg

log2pi = np.log(2*np.pi)

def logpdf(x,mu,cov=None,L=None):

    if L is None:
        L = linalg.jitchol(cov)

    Linv = np.linalg.inv(L)

    dim = x.shape[0]
    dev = x - mu
    maha = np.sum(np.square(np.dot(dev, Linv.T)), axis=-1)
    det = np.log(np.linalg.det(L))

    return -.5*(dim*log2pi + 2*det + maha)
