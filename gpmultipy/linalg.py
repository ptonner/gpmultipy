"""
This code was adapted from the GPy library.

Copyright (c) 2012, the GPy authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import numpy as np
from scipy import linalg
from scipy.linalg import lapack, blas
import logging, traceback

def jitchol(A, maxtries=5):
	A = np.ascontiguousarray(A)
	L, info = lapack.dpotrf(A, lower=1)
	if info == 0:
		return L
	else:
		diagA = np.diag(A)
		if np.any(diagA <= 0.):
			raise linalg.LinAlgError("not pd: non-positive diagonal elements")
		jitter = diagA.mean() * 1e-6
		num_tries = 1
		while num_tries <= maxtries and np.isfinite(jitter):
			try:
				L = linalg.cholesky(A + np.eye(A.shape[0]) * jitter, lower=True)

				if np.linalg.det(L) == 0:
					raise

				# logging.warning('\n'.join(['Added jitter of {:.10e}'.format(jitter),]))

				return L
			except:
				jitter *= 10
			finally:
				num_tries += 1
		raise linalg.LinAlgError("not positive definite, even with jitter.")
	import traceback
	try: raise
	except:
		logging.warning('\n'.join(['Added jitter of {:.10e}'.format(jitter),
			'  in '+traceback.format_list(traceback.extract_stack(limit=3)[-2:-1])[0][2:]]))
	return L

def invert_K(K):

    try:
        chol = jitchol(K)
        chol_inv = np.linalg.inv(chol)
    except np.linalg.linalg.LinAlgError,e:
        logger = logging.getLogger(__name__)
        logger.error('Kernel inversion error: %s'%str(self.parameters))
        raise(e)
    inv = np.dot(chol_inv.T,chol_inv)

    return inv
