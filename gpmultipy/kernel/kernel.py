import numpy as np
from .. import linalg
import logging

class Kernel(object):

    def __init__(self,p):
        self.p = p

    def K_inv(self,X,*args,**kwargs):

        K = self.K(X,*args,**kwargs)

        try:
            chol = linalg.jitchol(K)
            chol_inv = np.linalg.inv(chol)
        except np.linalg.linalg.LinAlgError,e:
            logger = logging.getLogger(__name__)
            # logger.error('Kernel inversion error: %s'%str(self.parameters))
            raise(e)
        inv = np.dot(chol_inv.T,chol_inv)

        return inv
