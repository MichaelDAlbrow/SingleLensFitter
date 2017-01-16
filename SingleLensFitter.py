import numpy as np
from scipy import linalg

import emcee
import george
from george import kernels

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from pylab import subplots_adjust

import corner
from astropy.stats import mad_std

class SingleLensFitter():

	"""Class definition for single microlens event fitter."""

	def __init__(self,data,initial_parameters=None,eigen_lightcurves=None):

		"""Initialise the SingleLensFitter.

		inputs:

			data:			A dictionary with each key being a data set name string and each 
						value being a tuple of (date, flux, flux_err) each being numpy
						arrays.

			initial_parameters:	A numpy array of starting guess values for u_0, t_0, t_E.

			eigen_lightcurves:   	If defined, this should be a dictionary with the same keys as data,
						and each value being an n x m numpy array of n lightcurves each 
						with m data values corresponding to the dates of the input data.
						These lightcurves (detrend vectors) are fitted linearly to the data
						at the same time as the magnification and flux parameters. An
						eigen_lightucrves dictionary entry can be defined for any subset 
						of data sources, and different data sources can have different
						numbers of eigenlightcurves.

		"""

		self.data = data
		self.eigen_lightcurves = eigen_lightcurves
		self.initial_parameters = initial_parameters

		self.marginalise_linear_parameters = True
		self.fit_blended = True

		self.u0_limits = (0.0,1.3)
		self.tE_limits = (0.5,200)
		self.t0_limits = None

		self.use_gaussian_process_model = False
		self.GP_default_params = (1.0,-2.0)
		self.ln_a_limits = (-5,15)
		self.ln_tau_limits = (-5.5, -1)
		self.ln_a = {}
		self.ln_tau = {}

		self.use_mixture_model = False
		self.mixture_default_params = (0.0001,1.0e8,0.0)
		self.P_b_limits = (0.0,0.05)
		self.V_b_limits = (0.0,1.0e12)
		self.Y_b_limits = (-1.e5,1.e5)
		self.P_b = {}
		self.V_b = {}
		self.Y_b = {}

		self.nwalkers = 50
		self.nsteps = 200
		self.thresh_std = 0.05
		self.thresh_mean = 0.01
		self.max_burnin_iterations = 20

		self.plotprefix = 'single_lens_fit'
		self.samples = None
		self.n_plot_samples = 30
		self.make_plots = True

		return


	def lnprior_GP(self,GP_params):
		params = ['ln_a','ln_tau']
		for i, p in enumerate(params):
			prange = eval('self.'+p+'_limits')
			if prange:
				if GP_params[i] < prange[0] or GP_params[i] > prange[1]:
					return -np.inf
		return 0.0

	

	def lnprior_mixture(self,mixture_params):
		params = ['P_b','V_b','Y_b']
		for i, p in enumerate(params):
			prange = eval('self.'+p+'_limits')
			if prange:
				if mixture_params[i] < prange[0] or mixture_params[i] > prange[1]:
					return -np.inf
		return 0.0

	

	def lnprior_ulens(self):
		params = ['u0','t0','tE']
		for p in params:
			prange = eval('self.'+p+'_limits')
			if prange:
				param = eval('self.'+p)
				if param < prange[0] or param > prange[1]:
					return -np.inf
		return 0.0


	def lnprob(self,p):

		self.u0 = p[0]
		self.t0 = p[1]
		self.tE = p[2]

		lp = self.lnprior_ulens()
		pi = 3

		for data_set_name in self.data.keys():

			if self.use_mixture_model:

				self.P_b[data_set_name] = p[pi]
				self.V_b[data_set_name] = p[pi+1]
				self.Y_b[data_set_name] = p[pi+2]
				lp += self.lnprior_mixture(p[pi:pi+3])
				pi += 3

			if self.use_gaussian_process_model:

				self.ln_a[data_set_name] = p[pi]
				self.ln_tau[data_set_name] = p[pi+1]
				lp += self.lnprior_GP(p[pi:pi+2])
				pi += 2

		if np.isfinite(lp):
			lp += self.lnlikelihood()
		else:
			return -np.inf

		return lp


	def lnlikelihood(self):

		lnprob = 0.0

		for data_set_name in self.data.keys():

			t, y, yerr = self.data[data_set_name]
			mag = self.magnification(t)

			if self.use_gaussian_process_model:
				a = np.exp(self.ln_a[data_set_name])
				tau = np.exp(self.ln_tau[data_set_name])
				gp = george.GP(a * kernels.ExpKernel(tau))
				gp.compute(t, yerr)
				self.cov = gp.get_matrix(t)
				result, lp = self.linear_fit(data_set_name,mag)
				model = self.compute_lightcurve(data_set_name,t)
				lnprob = gp.lnlikelihood(y-model)

			else:
				result, lp = self.linear_fit(data_set_name,mag)
				lnprob += lp

		return lnprob


	def linear_fit(self,data_key,mag):

		_, y, yerr = self.data[data_key]

		if self.use_gaussian_process_model:
			C = self.cov
		else:
			C = np.diag(yerr**2)

		C_inv = linalg.inv(C)

		if self.fit_blended:
			A = np.vstack((np.ones_like(mag),mag))
			n_params = 2
		else:
			A = (mag-1.0).reshape(1,len(mag))
			n_params = 1

		if self.eigen_lightcurves is not None:
			if data_key in self.eigen_lightcurves:
				eigs = self.eigen_lightcurves[data_key]
				for i in range(eigs.shape[0]):
					A = np.vstack((A,eigs[i,:]))
					n_params += 1

		S = np.dot(A,np.dot(C_inv,A.T)).reshape(n_params,n_params)
		b = np.dot(A, np.dot(C_inv,y).T)

		if self.marginalise_linear_parameters:

			try:
				M = linalg.inv(S)
			except linalg.LinAlgError:
				return (0,0), -np.inf

			g = np.dot(M,b)
			D = y - np.dot(A.T,g)
			chi2 = np.dot(D.T,np.dot(C_inv,D))

			if self.use_mixture_model:

				detM = linalg.det(M)
				lnprob = np.log( 2*np.pi*np.sum( (1.0 - self.P_b[data_key]) * \
						np.exp(-D**2/(2.0*yerr**2)) / \
						np.sqrt(detM) + \
						self.P_b[data_key]*np.exp(-(y-self.Y_b[data_key])**2 / \
						2*(self.V_b[data_key]+yerr**2)) / \
						np.sqrt(2*np.pi*(self.V_b[data_key]+yerr**2)) ))
			else:

				lnprob = np.log(2*np.pi) - 0.5*chi2 - 0.5*np.log(linalg.det(M))

			return g, lnprob

		else:

			try:
				a = linalg.solve(S,b)
			except linalg.LinAlgError:
				return (0,0), -np.inf
			D = y - np.dot(A.T,a)
			chi2 = np.dot(D.T,np.dot(C_inv,D))

			if self.use_mixture_model:

				lnprob = np.sum( np.log( (1.0 - self.P_b[data_key])*np.exp(-D**2/(2.0*yerr**2)) / \
						np.sqrt(2*np.pi*yerr**2) + \
						self.P_b[data_key]*np.exp(-(y-self.Y_b[data_key])**2 / \
						2*(self.V_b[data_key]+yerr**2)) / \
						np.sqrt(2*np.pi*(self.V_b[data_key]+yerr**2)) ))

			else:

				lnprob = -np.log(np.sum(np.sqrt(2*np.pi*yerr**2))) - 0.5*chi2

			return a, lnprob


	def magnification(self,t,params=None):

		if params is None:
			u0 = self.u0
			t0 = self.t0
			tE = self.tE
		else:
			u0, t0, tE = params[:3]

		tau = (t-t0)/tE
		u = np.sqrt(u0**2+tau**2)
		return (u**2+2)/(u*np.sqrt(u**2+4))


	def fit(self):
		
		if self.initial_parameters is None:
			raise Exception('Error in SingleLensFitter.fit(): No initial_parameters found.')
			return None

		print 'Initial parameters:', self.initial_parameters

		parameter_labels = [r"$u_0$",r"$t_0$",r"$t_E$"]
		testdim = [0,2]

		ndim = 3

		for data_set_name in self.data.keys():

			if self.use_mixture_model:

				self.initial_parameters.append(self.mixture_default_params[0])
				self.initial_parameters.append(self.mixture_default_params[1])
				self.initial_parameters.append(self.mixture_default_params[2])
				ndim += 3
				parameter_labels.append(data_set_name+'_P_b')
				parameter_labels.append(data_set_name+'_V_b')
				parameter_labels.append(data_set_name+'_Y_b')


			if self.use_gaussian_process_model:
				self.initial_parameters.append(self.GP_default_params[0])
				self.initial_parameters.append(self.GP_default_params[1])
				ndim += 2
				parameter_labels.append(data_set_name+'_ln_a')
				parameter_labels.append(data_set_name+'_ln_tau')


		p = [np.array(self.initial_parameters) + 1e-8 * np.random.randn(ndim) \
					for i in xrange(self.nwalkers)]

		sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.lnprob)

		print("Running burn-in...")

		iteration = 0

		pstd = np.random.randn(ndim)
		pstd_last = np.random.randn(ndim)
		pmean = np.random.randn(ndim)
		pmean_last = np.random.randn(ndim)
    
		while (np.max(np.abs(pstd[testdim]/pstd_last[testdim]-1)) > self.thresh_std) or \
			(np.max(np.abs(pmean[testdim]/pmean_last[testdim]-1)) > self.thresh_mean) \
			and (iteration < self.max_burnin_iterations):

			p, lnp , _ = sampler.run_mcmc(p, self.nsteps)

			pstd_last = pstd
			this_sampler = sampler.chain[:,-self.nsteps:,:].reshape(self.nwalkers*self.nsteps,ndim)
			pstd = mad_std(this_sampler,axis=0)
			pmean_last = pmean
			pmean = np.median(this_sampler,axis=0)
			print 'iteration:', iteration
			print 'pmean:', pmean
			print 'pstd:', pstd
        		iteration += 1

		if self.make_plots:
			self.plot_chain(sampler,suffix='-burnin.png',labels=parameter_labels)
			ind = 3

			npar = 0
			if self.use_mixture_model:
				npar += 3
			if self.use_gaussian_process_model:
				npar += 2

			if npar > 0:

				for data_set_name in self.data.keys():
				
					self.plot_chain(sampler,index=range(ind,npar+ind),  \
							suffix='-burnin-'+data_set_name+'.png', \
							labels=parameter_labels[ind:ind+npar])
					ind += npar

    		sampler.reset()

		print("Running production...")

		p0, lnp, _ = sampler.run_mcmc(p, self.nsteps)

		p = p0[np.argmax(lnp)]
		print p

		self.samples = sampler.flatchain

		u0_mcmc, t0_mcmc, tE_mcmc  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), \
							zip(*np.percentile(self.samples[:,:3], \
 							[16, 50, 84], axis=0)))

		self.u0 = u0_mcmc[0]
		self.t0 = t0_mcmc[0]
		self.tE = tE_mcmc[0]

		if self.make_plots:

			self.plot_chain(sampler,suffix='-final.png',labels=parameter_labels)
			ind = 3

			if npar > 0:

				for data_set_name in self.data.keys():
				
					self.plot_chain(sampler,index=range(ind,npar+ind),  \
							suffix='-final-'+data_set_name+'.png', \
							labels=parameter_labels[ind:ind+npar])
					ind += npar

			self.plot_lightcurves()
			self.plot_chain_corner()

		print 'Results:'
		print 'u0', u0_mcmc
		print 't0', t0_mcmc
		print 'tE', tE_mcmc

		return


	def plot_chain(self,s,index=None,suffix='',labels=[r"$u_0$",r"$t_0$",r"$t_E$"]):

		if index is None:
			index = [0,1,2]

		ndim = len(index)

		plt.figure()
		
		subplots_adjust(hspace=0.0001)

		for i in range(ndim):

			if i == 0:
				plt.subplot(ndim,1,i+1)
				ax1 = plt.gca()
			else:
				plt.subplot(ndim,1,i+1,sharex=ax1)

			plt.plot(s.chain[:,:,index[i]].T, '-', color='k', alpha=0.3)

			if labels:
				plt.ylabel(labels[i])

		ax = plt.gca()

		if i < ndim-1:
			ax.axes.xaxis.set_ticklabels([])
			ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))
			ax.locator_params(axis='y',nbins=4)

		plt.savefig(self.plotprefix+suffix)
		plt.close()


	def compute_lightcurve(self,data_key, x, params=None):

		t, _, _ = self.data[data_key]
		coeffs, _ = self.linear_fit(data_key,self.magnification(t,params))

		if self.fit_blended:
			fx = coeffs[0]+coeffs[1]*self.magnification(x,params)
		else:
			fx = coeffs[0]*(self.magnification(x,params)-1)

		return fx



	def plot_lightcurves(self):

		plt.figure()
    
		colour = iter(plt.cm.jet(np.linspace(0,1,len(self.data))))

		xmin = self.initial_parameters[1]-2*self.initial_parameters[2]
		xmax = self.initial_parameters[1]+2*self.initial_parameters[2]

		n_data = len(self.data)
		for i, data_set_name in enumerate(self.data.keys()):

			t, y, yerr = self.data[data_set_name]
			c=next(colour)

			if i == 0:
				plt.subplot(n_data,1,i+1)
				ax1 = plt.gca()
			else:
				plt.subplot(n_data,1,i+1,sharex=ax1)

			y_cond = y
			if self.eigen_lightcurves is not None:
				if data_set_name in self.eigen_lightcurves:
					coeffs, _ = self.linear_fit(data_set_name,self.magnification(t))
					ci = 1
					if self.fit_blended:
						ci = 2
					eigs = self.eigen_lightcurves[data_set_name]
					for j in range(eigs.shape[0]):
						y_cond -= coeffs[ci+j]*eigs[j,:]

			plt.errorbar(t, y_cond, yerr=yerr, fmt=".", color=c, capsize=0)
			ax=plt.gca()
			ax.set_xlim(xmin,xmax)
			plt.xlabel(r"$\Delta t (d)$")
			plt.ylabel(data_set_name+r"  $\Delta F$")
    
			x = np.linspace(xmin,xmax, 3000)
			plt.plot(x, self.compute_lightcurve(data_set_name,x),color="k")
        		ylim = ax.get_ylim()
        
        		# Plot posterior samples.
			for s in self.samples[np.random.randint(len(self.samples), size=self.n_plot_samples)]:

				if self.use_gaussian_process_model:

					# Set up the GP for this sample.
					a, tau = np.exp(s[3+2*i:3+2*i+2])
					gp = george.GP(a * kernels.ExpKernel(tau))
					gp.compute(t, yerr)
					cov = gp.get_matrix(t)
					modelt = self.compute_lightcurve(data_set_name,t,params=s)
					modelx = self.compute_lightcurve(data_set_name,x,params=s)

					# Compute the prediction conditioned on the observations
					# and plot it.
					m = gp.sample_conditional(y - modelt,x) + modelx
					plt.plot(x, m, color="#4682b4", alpha=0.3)

				else:

					plt.plot(x, self.compute_lightcurve(data_set_name,x,params=s), \
							color="#4682b4",alpha=0.3)

			ax.set_ylim(ylim)

		plt.savefig(self.plotprefix+'-lc.png')

		plt.close()

 
	def plot_chain_corner(self):

		figure = corner.corner(self.samples[:,:3],
					labels=[r"$u_0$",r"$t_0$",r"$t_E$"],
					quantiles=[0.16, 0.5, 0.84],
					truths=(self.u0,self.t0,self.tE),
					show_titles=True, title_args={"fontsize": 12})
		figure.savefig(self.plotprefix+'-pdist.png')






		 	
