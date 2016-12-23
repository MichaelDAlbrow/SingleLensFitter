import numpy as np
from scipy import linalg
import emcee
import matplotlib.pyplot as plt
import corner


class SingleLensFitter():

	"""Class definition for single microlens event fitter."""

	def __init__(self,data,initial_parameters=None):

		"""Initialise the SingleLensFitter.

		input data:	a dictionary with each key being a data set name and each value being
				a tuple of (date,flux,flux_err,reference_flux) with the first three of 
				these being numpy arrays
		"""

		self._data = data
		self._eigen_lightcurves = None
		self._marginalise_linear_parameters = True
		self._fit_blended = True
		self._u0_limits = (0.0,1.3)
		self._tE_limits = (0.5,200)
		self._t0_limits = None
		self._initial_parameters = initial_parameters
		self._nwalkers = 50
		self._nsteps = 200
		self._thresh_std = 0.05
		self._thresh_mean = 0.01
		self._max_burnin = 20
		self._plotprefix = 'single_lens_fit'

		return


	def get(self,parameter):
		try:
			value = getattr(self,'_'+parameter)
		except AttributeError:
			value = None
		return value


	def set(self,parameter,value):
		try:
			value = getattr(self,'_'+parameter)
		except AttributeError:
			print 'Error: '+parameter+'is not a defined parameter'
			sys.exit(0)
		setattr(self,'_'+parameter,value)


	def print_parameters(self,parameter=None):
		if parameter:
			try:
				value = getattr(self,'_'+parameter)
			except AttributeError:
				print 'Error: '+parameter+'is not a defined parameter'
		else:
			print 'Parameters:'
			for par in dir(self):
				print par, getattr(self,'_'+par)
			print


	def lnprior_ulens(self):
		for p in ['u0','t0','tE']:
			prange = eval('self._'+p+'_limits')
			if prange:
				param = eval('self._'+p)
				if param < prange[0] or param > prange[1]:
					return -np.inf
		return 0.0


	def lnprob(self):
		lp = self.lnprior_ulens()
		if np.isfinite(lp):
			lp += self.lnlikelihood()
			return lp
		return -np.inf


	def lnlikelihood(self):
		lnprob = 0.0
		for data_set_name in self._data.keys():
			t, y, yerr, _ = self._data[data_set_name]
			mag = self.magnification(t).T
			result, lp = self.linear_fit(y,yerr,mag)
			lnprob += lp
		return lnprob


	def linear_fit(self,y,err,mag):

		err_inv = 1.0/err

		if self._fit_blended:
			A = np.vstack((1*err_inv,mag*err_inv))
		else:
			A = (mag-1.0)*err_inv

		if self._eigen_lightcurves is not None:
			for eig in self.eigen_lightcurves:
				A = np.vstack(A,eig*err_inv)

		S = np.dot(A,A.T)
		b = np.dot(A, y*err_inv)

		if self._marginalise_linear_parameters:

			try:
				M = linalg.inv(S)
			except LinAlgError:
				return (0,0), -np.inf

			g = np.dot(M,b)
			D = y*err_inv - np.dot(A.T,g)
			chi2 = np.dot(D.T,D)
			lnprob = np.log(2*np.pi) - 0.5*chi2 - 0.5*np.log(linalg.det(M))
			return g, lnprob

		else:

			try:
				a = np.linalg.solve(S,b)
			except LinAlgError:
				return (0,0), -np.inf
			return a, lnprob


	def magnification(self,t):
		tau = (t-self.t0)/self.tE
		u = np.sqrt(self.u0**2+tau**2)
		return (u**2+2)/(u*np.sqrt(u**2+4))


	def fit(self):
		
		if self._initial_parameters is None:
			print 'Error in SingleLensFitter.fit(): No initial_parameters found. Exiting...'
			return None

		ndim = 3
		testdim = [0,2]

		p = [np.array(self._initial_parameters) + 1e-8 * np.random.randn(ndim) for i in xrange(nwalkers)]

		sampler = emcee.EnsembleSampler(self._nwalkers, ndim, self.lnprob)

		print("Running burn-in...")

		iteration = 0

		pstd = np.random.randn(ndim)
		pstd_last = np.random.randn(ndim)
		pmean = np.random.randn(ndim)
		pmean_last = np.random.randn(ndim)
    
		while (np.max(np.abs(pstd[testdim]/pstd_last[testdim]-1)) > thresh_std) or (np.max(np.abs(pmean[testdim]/pmean_last[testdim]-1)) > thresh_mean) and (iteration < max_burnin):
			p, lnp , _ = sampler.run_mcmc(p, nsteps)
			pstd_last = pstd
			this_sampler = sampler.chain[:,-nsteps:,:].reshape(nwalkers*nsteps,ndim)
			pstd = mad_std(this_sampler,axis=0)
			pmean_last = pmean
			pmean = np.median(this_sampler,axis=0)
			print 'iteration:', iteration
			print 'pmean:', pmean
			print 'pstd:', pstd
        		iteration += 1

		self.plotchain(sampler,plotfile=self._plotprefix+'-burnin.png',labels=self._data.keys())

    		sampler.reset()






		 	
