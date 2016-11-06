from interval import Interval
import numpy as np

class FunctionInterval(Interval):

	def __init__(self,samples,alpha,start=1e-6,tol=1e-6,maxiter=100):
		Interval.__init__(self,samples,alpha,ndim=2)

		p = samples.shape[0]
		alphaPotential = 1.*np.arange(self.n)/self.n
		self.alpha = filter(lambda x: abs(x-self.alpha)==abs(self.alpha-alphaPotential).min(),alphaPotential)[0]

		self.mean = samples.mean(0)
		self.std = samples.std(0)
		self.lb,self.ub = self.mean-2*self.std,self.mean+2*self.std

		self.epsilon = start

		# function to calculate the empirical interval alpha for a given epsilon, x
		check = lambda x: 1.*sum((self.samples>self.lb-x).all(1) & (self.samples<self.ub+x).all(1))/self.n

		bounds = []

		# double to find lower and upper epislon bound
		while check(self.epsilon)<self.alpha:
			bounds.append((self.epsilon,check(self.epsilon)))
			self.epsilon *= 2

		bounds.append((self.epsilon,check(self.epsilon)))

		# binary search
		eLb,eUb = self.epsilon/2,self.epsilon
		i = 0
		while True:
			if abs(check(eLb)-alpha)<tol:
				self.epsilon = eLb
				break
			elif abs(check(eUb)-alpha)<tol:
				self.epsilon = eUb
				break

			nb = (eLb + eUb)/2

			if check(nb)<alpha:
				eLb = nb
				bounds.append((nb,check(nb)))
			else:
				eUb = nb
				bounds.append((nb,check(nb)))

			i+=1
			if i > maxiter:
				self.epsilon = eLb
				break

		self.bounds = bounds

		self.lb = self.mean - self.std - self.epsilon
		self.ub = self.mean + self.std + self.epsilon

		#return epsilon,bounds

	def plot(self,x=None,alpha=.2,c='b'):
		import matplotlib.pyplot as plt

		if x is None:
			x = np.arange(self.mean.shape[0])

		plt.plot(x,self.mean,color=c)
		plt.fill_between(x,self.lb,self.ub,alpha=alpha,color=c)

	def contains(self,x):
		return (x>self.lb).all() & (x<self.ub).all()


def functionInterval(samples,start=1e-6,alpha=.95,tol=1e-6,maxiter=100):

	# change alpha to be best possible given the number of samples
	p = samples.shape[0]
	alphaPotential = 1.*np.arange(p)/p
	alpha = filter(lambda x: abs(x-alpha)==abs(alpha-alphaPotential).min(),alphaPotential)[0]

	mean = samples.mean(0)
	std = samples.std(0)
	lb,ub = mean-2*std,mean+2*std

	epsilon = start

	# function to calculate the empirical interval alpha for a given epsilon, x
	check = lambda x: 1.*sum((samples>lb-x).all(1) & (samples<ub+x).all(1))/samples.shape[0]

	bounds = []

	# double to find lower and upper epislon bound
	while check(epsilon)<alpha:
		bounds.append((epsilon,check(epsilon)))
		epsilon *= 2

	bounds.append((epsilon,check(epsilon)))

	# binary search
	eLb,eUb = epsilon/2,epsilon
	i = 0
	while True:
		if abs(check(eLb)-alpha)<tol:
			epsilon = eLb
			break
		elif abs(check(eUb)-alpha)<tol:
			epsilon = eUb
			break

		nb = (eLb + eUb)/2

		if check(nb)<alpha:
			eLb = nb
			bounds.append((nb,check(nb)))
		else:
			eUb = nb
			bounds.append((nb,check(nb)))

		i+=1
		if i > maxiter:
			epsilon = eLb
			break

	return epsilon,bounds
