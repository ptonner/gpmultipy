import numpy as np

class Interval(object):

	def __init__(self,samples,alpha,ndim=1):
		self.samples = samples
		self.n = len(self.samples)

		# DOES ALPHA MEAN INCLUSIVE OR EXCLUSIVE PROB????
		self.alpha = alpha

		if self.alpha < 0 or self.alpha > 1:
			raise ValueError("must provide alpha between 0 and 1")

		# if self.samples.ndim != ndim:
		# 	raise ValueError("sample dimensions does not match %d"%ndim)

	def contains(self,x):
		raise NotImplemented("implement this for your interval!")
