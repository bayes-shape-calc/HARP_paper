import numpy as np
import numba as nb

from math import lgamma
@nb.vectorize(cache=True)
def gammaln(x):
	return lgamma(x)

@nb.vectorize(cache=True)
def psi(x):
	''' This is the Cephes version used in Scipy, but I rewrote it in Python'''
	A = [
		8.33333333333333333333E-2,
		-2.10927960927960927961E-2,
		7.57575757575757575758E-3,
		-4.16666666666666666667E-3,
		3.96825396825396825397E-3,
		-8.33333333333333333333E-3,
		8.33333333333333333333E-2
	]
	if np.isnan(x):
		raise Exception('nan in psi')

	# check for positive integer up to 10
	if x <= 10. and x==np.floor(x):
		y = 0.0
		for i in range(1,np.floor(x)):
			y += 1./i
		y -= 0.577215664901532860606512090082402431 #Euler
		return y
	else:
		s = x
		w = 0.0
		while s < 10.:
			w += 1./s
			s += 1.
		z = 1.0 / (s * s);


		poly = A[0]
		for aa in A[1:]:
			poly = poly*z +aa
		y = z * poly
		y = np.log(s) - (0.5 / s) - y - w;

		return y

@nb.vectorize(cache=True)#([nb.float64(nb.float64)],cache=True)
def trigamma(q):
	''' This is the Cephes version of zeta used in Scipy, but I rewrote it in Python for x = 2'''

	A = [
		12.0,
		-720.0,
		30240.0,
		-1209600.0,
		47900160.0,
		-1.8924375803183791606e9,
		7.47242496e10,
		-2.950130727918164224e12,
		1.1646782814350067249e14,
		-4.5979787224074726105e15,
		1.8152105401943546773e17,
		-7.1661652561756670113e18
	]
	macheps =  2.22044604925e-16 ## double for numpy

	if np.isnan(q):
		raise Exception('nan in trigamma')

	if q <= 0.0:
		if q == np.floor(q):
			return np.nan
		return np.inf

	# /* Asymptotic expansion
	#  * https://dlmf.nist.gov/25.11#E43
	#  */
	if (q > 1e8):
		return (1. + .5/q) /q

	# /* Euler-Maclaurin summation formula */
	# /* Permit negative q but continue sum until n+q > +9 .
	#  * This case should be handled by a reflection formula.
	#  * If q<0 and x is an integer, there is a relation to
	#  * the polyGamma function.
	#  */
	s = q**(-2.)
	a = q
	i = 0
	b = 0.0

	while ((i<9) or (a<=9.0)):
		i+= 1
		a += 1.0
		b = a**(-2.)
		s += b
		if np.abs(b/s) < macheps:
			return s

	w = a
	s += b*w
	s -= .5*b
	a = 1.
	k = 0.
	for i in range(12):
		a *= 2+k
		b /= w
		t = a*b / A[i]
		s = s +t
		t = np.abs(t/s)
		if t < macheps:
			return s
		k += 1.0
		a *= 2 + k
		b /= w
		k += 1.
	return s


@nb.vectorize
def invert_psi(y):
	## Minka appendix C -- doesn't really work if x is negative....

	## initial guess (Minka 149)
	if y >= -2.22:
		x = np.exp(y)+.5
	else:
		x = -1./(y - psi(1.))

	## iterations
	for i in range(5): ## Minka says only 5 to get 14 bit accuracy
		x = x - (psi(x)-y)/trigamma(x) ## Minka 146
	return x

################################################################################
################################################################################
################################################################################

@nb.njit
def inv_jacobian(invJ,invE,J,theta,M,Nm,lnpm,lnqm):
	alphas = theta[0:M]
	betas = theta[M:2*M]
	mu_a,mu_b,tau_a,tau_b = theta[2*M:2*M+4]

	for i in range(M):
		Ni = Nm[i]
		a = alphas[i]
		b = betas[i]
		lnp = lnpm[i]
		lnq = lnqm[i]
		tab = trigamma(a+b)

		J[i,i]     = Ni*tab - Ni*trigamma(a) + (1.-mu_a*tau_a)/(a*a) - tau_a/(a*a)*(1. - np.log(a)) ## aa
		J[M+i,M+i] = Ni*tab - Ni*trigamma(b) + (1.-mu_b*tau_b)/(b*b) - tau_b/(b*b)*(1. - np.log(b)) ## bb
		J[M+i,i]   = Ni*tab ## ba
		J[i,M+i]   = Ni*tab ## ab

		J[2*M,i]     = tau_a/a ## ma a
		J[2*M+1,M+i] = tau_b/b ## mb b
		J[2*M+2,i] = (mu_a-np.log(a))/a ## ta a
		J[2*M+3,M+i] = (mu_b-np.log(b))/b ## tb b
		J[i,2*M] = tau_a/a ## a ma
		J[M+i,2*M+1] = tau_b/b ## b mb
		J[i,2*M+2] = (mu_a-np.log(a))/a ## a ta
		J[M+i,2*M+3] = (mu_b-np.log(b))/b ## b tb

	J[2*M,2*M] = -M*tau_a ## ma ma
	J[2*M+1,2*M+1] = -M*tau_b ## mb mb
	J[2*M,2*M+2] = np.sum(np.log(alphas)) -M*mu_a ## ma ta
	J[2*M+1,2*M+3] = np.sum(np.log(betas)) -M*mu_b ## mb tb
	J[2*M+2,2*M] = J[2*M,2*M+2] ## ta ma
	J[2*M+3,2*M+1] = J[2*M+1,2*M+3] ## tb mb
	J[2*M+2,2*M+2] = -.5*(M-2.)/(tau_a*tau_a) ## ta ta
	J[2*M+3,2*M+3] = -.5*(M-2.)/(tau_b*tau_b) ## tb tb

	## invert
	for i in range(M):
		det = J[i,i]*J[M+i,M+i]-J[i,M+i]*J[M+i,i]
		invE[i,i] = J[M+i,M+i]/det
		invE[M+i,M+i] = J[i,i]/det
		invE[i,M+i] = -J[i,M+i]/det
		invE[M+i,i] = -J[M+i,i]/det
	H = np.copy(J[2*M:2*M+4,2*M:2*M+4])
	F = np.copy(J[0:2*M,2*M:2*M+4])
	G = np.copy(J[2*M:2*M+4,0:2*M])

	HmGinvEF = H-np.dot(G,np.dot(invE,F))
	invHmGinvEF = np.linalg.inv(HmGinvEF)

	invEdF = np.dot(invE,F)
	GdinvE = np.dot(G,invE)
	invJ[:2*M,:2*M] = invE + np.dot(invEdF,np.dot(invHmGinvEF,GdinvE))
	invJ[0:2*M,2*M:2*M+4] = -np.dot(invEdF,invHmGinvEF)
	invJ[2*M:2*M+4,0:2*M] = -np.dot(invHmGinvEF,GdinvE)
	invJ[2*M:2*M+4,2*M:2*M+4] = invHmGinvEF
	return invJ

@nb.njit
def fxn(F,theta,M,Nm,lnpm,lnqm):
	alphas = theta[0:M]
	betas = theta[M:2*M]
	mu_a,mu_b,tau_a,tau_b = theta[2*M:2*M+4]

	for i in range(M):
		Ni = Nm[i]
		a = alphas[i]
		b = betas[i]
		lnp = lnpm[i]
		lnq = lnqm[i]
		F[i  ] = lnp + Ni*psi(a+b) - Ni*psi(a) - (1.-mu_a*tau_a + tau_a*np.log(a))/a
		F[M+i] = lnq + Ni*psi(a+b) - Ni*psi(b) - (1.-mu_b*tau_b + tau_b*np.log(b))/b

		if np.isnan(F[i]) or np.isnan(F[M+i]):
			raise Exception('Nan in fxn loop')

	la = np.log(alphas)
	lb = np.log(betas)
	sla = np.sum(la)
	slb = np.sum(lb)
	sla2 = np.sum(la*la)
	slb2 = np.sum(lb*lb)
	F[2*M+0] = tau_a*sla - M*tau_a*mu_a
	F[2*M+1] = tau_b*slb - M*tau_b*mu_b
	F[2*M+2] = .5*(M-2.)/tau_a - .5*sla2 + mu_a*sla - M/2.*(mu_a**2.)
	F[2*M+3] = .5*(M-2.)/tau_b - .5*slb2  + mu_b*slb  - M/2.*(mu_b**2.)

	if np.any(np.isnan(F)):
		raise Exception('Nan in fxn')
	return F

@nb.njit
def lnJ(theta,M,Nm,lnpm,lnqm):
	alphas = theta[0:M]
	betas = theta[M:2*M]
	mu_a,mu_b,tau_a,tau_b = theta[2*M:2*M+4]

	out = .5*(float(M)-2.)*np.log(tau_a) + .5*(float(M)-2.)*np.log(tau_b) - M*np.log(2.*np.pi) - .5*M*tau_a*(mu_a**2.) - .5*M*tau_b*(mu_b**2.)
	for i in range(M):
		Ni = Nm[i]
		a = alphas[i]
		b = betas[i]
		lnp = lnpm[i]
		lnq = lnqm[i]

		out += (a-1.)*lnp  +(b-1.)*lnq + Ni*gammaln(a+b) - Ni*gammaln(a) - Ni*gammaln(b)
		out += -np.log(a) -.5*tau_a*(np.log(a))**2. + tau_a*mu_a*np.log(a) - np.log(b) -.5*tau_b*(np.log(b))**2. + tau_b*mu_b*np.log(b)

		if np.isnan(out):
			print(mu_a,mu_b,tau_a,tau_b,i,Ni,a,b,lnp,lnq)
			print(alphas)
			print(betas)
			raise Exception('Nan in lnJ loop')
	return out


@nb.njit
def fast_stats(x):
	ceps = 1e-6
	n = x.size
	lns = np.zeros(2,dtype=nb.double)
	for i in range(n):
		if x[i] < ceps:
			lns[0] += np.log(ceps)
			lns[1] += np.log(1.-ceps)
		elif x[i] > 1.-ceps:
			lns[0] += np.log(1.-ceps)
			lns[1] += np.log(ceps)
		else:
			lns[0] += np.log(x[i])
			lns[1] += np.log(1.-x[i])
	return lns

@nb.njit
def initialize(M,Nm,lnpm,lnqm):
	### initialize
	s = 10.**(np.random.rand()*4.-2)
	alphas = np.zeros(M)
	betas = np.zeros(M)
	psis = psi(s)
	for i in range(M):
		alphas[i] = invert_psi(lnpm[i]/Nm[i]+psis)
		betas[i] = invert_psi(lnqm[i]/Nm[i]+psis)
		if alphas[i] > 1e2:
			alphas[i] = 1e2
		if alphas[i] < 1e-2:
			alphas[i] = 1e-2
		if betas[i] > 1e2:
			betas[i] = 1e2
		if betas[i] < 1e-2:
			betas[i] = 1e-2
	for _ in range(5):
		for i in range(M):
			Fa = lnpm[i]/Nm[i] - psi(alphas[i]) + psi(alphas[i]+betas[i])
			Fb = lnqm[i]/Nm[i] - psi(betas[i]) + psi(alphas[i]+betas[i])
			tab = trigamma(alphas[i]+betas[i])
			Ja = tab - trigamma(alphas[i])
			Jb = tab
			Jc = tab
			Jd = tab - trigamma(betas[i])
			det = Ja*Jd-Jb*Jc
			alphas[i] -= (Jd*Fa-Jb*Fb)/det
			betas[i] -= (Jd*Fa-Jb*Fb)/det
			if alphas[i] > 1e2:
				alphas[i] = 1e2
			if alphas[i] < 1e-2:
				alphas[i] = 1e-2
			if betas[i] > 1e2:
				betas[i] = 1e2
			if betas[i] < 1e-2:
				betas[i] = 1e-2
	return alphas,betas

@nb.njit
def outerloop(theta,M,Nm,lnpm,lnqm,maxiter,stoptoobig):
	l0 = lnJ(theta,M,Nm,lnpm,lnqm)
	max_theta = np.copy(theta)
	max_lJ = -np.inf#l0
	# maxiter = 10000

	F0 = np.zeros((theta.size))
	F1 = np.zeros((theta.size))
	invJ = np.zeros((theta.size,theta.size))
	invE = np.zeros((2*M,2*M))
	J = np.zeros_like(invJ)

	for iteration in range(maxiter):
		invJ = inv_jacobian(invJ,invE,J,theta,M,Nm,lnpm,lnqm)
		F0 = fxn(F0,theta,M,Nm,lnpm,lnqm)
		jump = np.dot(invJ,F0)

		##
		f0 = .5*np.sum(F0**2.)
		lam = 1.

		for _ in range(2): ## limits the size of the jump. only needs two passes 
			theta_new = theta - lam*jump

			## Hard limits from priors -- outside=randomize. technically not true for alphas and betas
			if theta_new[-4] < -6 or theta_new[-4] > 6:
				theta_new[-4] = np.random.rand()*12-6
			if theta_new[-3] < -6 or theta_new[-3] > 6:
				theta_new[-3] = np.random.rand()*12-6

			if theta_new[-2] < 1e-4 or theta_new[-2] > 1e4:
				theta_new[-2] = 10.**(np.random.rand()*8-4)
			if theta_new[-1] < 1e-4 or theta_new[-1] > 1e4:
				theta_new[-1] = 10.**(np.random.rand()*8-4)

			for i in range(2*M):
				if theta_new[i] > 1e4:
					theta_new[i] = 1e4
				if theta_new[i] < 1e-4:
					theta_new[i] = 1e-4

			F1 = fxn(F1,theta_new,M,Nm,lnpm,lnqm)
			f1 = .5*np.sum(F1**2.)

			if f1<f0:
				break
			else:
				slope = np.sum(F0*jump)
				dlam = -.5*slope/(f1-f0-slope)
				if dlam < .1:
					dlam = .1
				lam *= dlam
				# print(lam)


		# l1 = f1
		l1 = lnJ(theta_new,M,Nm,lnpm,lnqm)

		## keep track of the best
		if l1 > max_lJ:
			max_lJ = l1
			max_theta = np.copy(theta_new)

		## check for convergence
		if iteration > 2:
			if np.abs(l1-l0)/np.abs(l0) < 1e-10:
				break
			if stoptoobig: ## if things start going very badly
				if np.any(theta[-2:] > 500):
					break
				
		theta = np.copy(theta_new)
		l0 = l1
		
	return iteration,max_lJ,max_theta

def full_infer(p_ress,maxiter=1000,initiate=None,stoptoobig=False):
	#### Infer hierarchical model of several beta distributions with different a,b sets
	#### infer alphas,betas, <ln alpha>, precision(ln alpha), <ln beta>, precision(ln beta)
	## P_ress = {P_res_m}
	## P_res_m ~ Beta(alpha_m,beta_m)
	## alpha_m ~ lognormal(mu_a,tau_a^-1), beta_m similar
	## Use newton raphson

	## input: p_ress is list of p_res arrays from blobBIS

	## Setup
	eps = np.finfo(np.double).eps
	M = len(p_ress)
	Nm = np.zeros(M,dtype='int')
	lnpm = np.zeros(M)
	lnqm = np.zeros(M)
	ceps = 1e-6
	for i in range(M):
		Nm[i] = p_ress[i].size
		lnpm[i],lnqm[i] = fast_stats(p_ress[i])

	alphas,betas = initialize(M,Nm,lnpm,lnqm)

	if initiate is None:
		mu_a = np.mean(np.log(alphas))
		mu_b = np.mean(np.log(betas))
		mu_a2 = np.mean(np.log(alphas)**2.)
		mu_b2 = np.mean(np.log(betas)**2.)
		tau_a = 1./(mu_a2 - mu_a**2.)
		tau_b = 1./(mu_b2 - mu_b**2.)
	
		if tau_a <= 1e-8:
			tau_a = 1e-8
		elif tau_a >= 1e4:
			tau_a = 1e4
		if tau_b <= 1e-8:
			tau_b = 1e-8
		elif tau_b >= 1e4:
			tau_b = 1e4
	else:
		mu_a,mu_b,tau_a,tau_b = initiate
		
	# mu_a = np.random.uniform(low=-1,high=1)
	# mu_b = np.random.uniform(low=-1,high=1)
	# tau_a = 10.**np.random.uniform(low=-1,high=1)
	# tau_b = 10.**np.random.uniform(low=-1,high=1)


	theta = np.concatenate([alphas,betas,[mu_a,mu_b,tau_a,tau_b]])

	iteration, max_lJ, max_theta = outerloop(theta,M,Nm,lnpm,lnqm,maxiter,stoptoobig=stoptoobig)

	# if iteration >= 999:
	# 	print(iteration,999)
	# 	import warnings
	# 	warnings.warn('did not converge within maxiterations')
	# 	# raise Exception('did not converge within maxiterations')

	invJ = np.zeros((max_theta.size,max_theta.size))
	invE = np.zeros((2*M,2*M))
	J = np.zeros_like(invJ)
	cov = -inv_jacobian(invJ,invE,J,max_theta,M,Nm,lnpm,lnqm)

	return iteration, max_lJ, max_theta, cov


def run_infer_restarts(p_ress,maxiter=1000,nres=10,initiate=None,stoptoobig=False):
	M = float(len(p_ress))
	iteration, lJ, theta0, cov = full_infer(p_ress,maxiter=maxiter,initiate=initiate,stoptoobig=stoptoobig)
	max_theta = theta0.copy()
	max_lJ = lJ
	max_cov = cov.copy()
	print("% 18.10f [% 10.5f % 10.5f % 10.5f % 10.5f] %d"%(lJ/M,theta0[-4],theta0[-3],theta0[-2],theta0[-1],iteration))
	for _ in range(nres-1):
		iteration,lJ, theta, cov = full_infer(p_ress,maxiter=maxiter,initiate=initiate,stoptoobig=stoptoobig)
		print("% 18.10f [% 10.5f % 10.5f % 10.5f % 10.5f] %d"%(lJ/M,theta[-4],theta[-3],theta[-2],theta[-1],iteration))
		if lJ > max_lJ:
			max_lJ = lJ
			max_theta = theta.copy()
			max_cov = cov.copy()
	return max_theta, max_cov, max_lJ/M

################################################################################
################################################################################
################################################################################

def hard_inf(p,threshold=.5,alpha0=1.,beta0=1.):
	#### solve <p> in bernoulli trials using count data
	## Counts x = p>threshold
	## L = \prod <p>^x (1-<p>)^(1-x)
	## prior = Beta(<p> | alpha0=1, beta0=1)
	## conjugate, so posterior is Beta(<p> | alpha = alpha0+sum(x), beta = beta0+sum(1-x))
	## return

	x = p >= threshold
	a = np.array(alpha0+np.sum(x))
	b = np.array(beta0+np.sum(1.-x))
	return np.array((a,b))

def beta_stats(a,b):
	from scipy.special import betaincinv
	out = betaincinv(a,b,[.025,.5,.975]) ## median
	out[1] = a/(a+b) ## mean
	return out

@nb.njit
def beta_infer(x):
	#### Solve dlnJ/da = 0 for s=a+b, a/s=<x>
	## use Newton Raphson s_{t+1}=s_t - F/(dF/ds)
	## F = psi(<x>s)-psi(s)-<lnx> = 0
	## dF/ds = 1/<x> * trigama(<x>s) - trigamma(s)
	##
	ceps = np.sqrt(np.finfo(np.double).eps)
	xx = x.copy()
	xx[xx<ceps] = ceps
	xx[xx>1.-ceps] = 1.-ceps
	Ex = np.nanmean(xx)
	Elnx = np.nanmean(np.log(xx))
	s = 1.

	s0 = s
	for iteration in range(1000):
		jump = (psi(Ex*s)-psi(s)-Elnx)/(trigamma(Ex*s)/Ex-trigamma(s))
		s1 = s-jump
		if iteration > 2:
			if np.abs(s1-s0)/np.abs(s0)<1e-10:
				break
		s0 = s1
		# print(iteration,s0)
	a = Ex*s
	b = s-a
	return np.array((a,b))


################################################################################
################################################################################
################################################################################

#### M1
def M1(pmr,alpha0=1.,beta0=1.):
	m1_theta = hard_inf(np.concatenate(pmr),threshold=.5,alpha0=alpha0,beta0=beta0)
	m1_ci = beta_stats(*m1_theta)
	return m1_theta,m1_ci

#### M2
def M2(pmr,alpha0=1.,beta0=1.):
	M = len(pmr)
	m2_thetas = np.zeros((M,2))
	m2_cis = np.zeros((M,3))
	for i in range(M):
		m2_thetas[i] = hard_inf(pmr[i],threshold=.5,alpha0=alpha0,beta0=beta0)
		m2_cis[i] = beta_stats(*m2_thetas[i])
	return m2_thetas,m2_cis

#### M4
def M4(pmr,maxiter=1000,nres=5,initiate=None,stoptoobig=False):
	theta,cov,lnj = run_infer_restarts(pmr,maxiter=maxiter,nres=nres,initiate=initiate,stoptoobig=stoptoobig)
	M = (theta.size-4)//2
	alphas = theta[:M]
	betas = theta[M:2*M]
	mu_a, mu_b = theta[-4:-2]
	tau_a, tau_b = theta[-2:]
	m4_cis = np.zeros((M,3))
	for i in range(M):
		m4_cis[i] = beta_stats(alphas[i],betas[i])

	marginal_cov = cov[-4:,-4:] ## marginal cf bishop 2.98 ie regardless of alphas, betas...
	return alphas,betas,mu_a,mu_b,tau_a,tau_b,m4_cis,marginal_cov,lnj

def process_sets_hierarchical(p_ress,unique_name='',initiates=None,maxiter=1000,nres=5,overwrite=False,stoptoobig=False):
	import os
	import h5py

	if os.path.exists(unique_name) and not overwrite:
		ps = []
		with h5py.File(unique_name,'r') as f:
			for i in range(len(p_ress)):
				ps.append(f['ps/%d'%(i)][:])
			mu_as = f['mu_as'][:]
			mu_bs = f['mu_bs'][:]
			tau_as = f['tau_as'][:]
			tau_bs = f['tau_bs'][:]
			covs = f['covs'][:]
		print('LOADED',unique_name)
	else:
		if overwrite:
			print('OVERWRITING',unique_name)
		alphass = []
		betass = []
		mu_as = []
		mu_bs = []
		tau_as = []
		tau_bs = []
		ps = []
		covs = []
		import time
		for i in range(len(p_ress)):
			pmr = p_ress[i]
			print('Running',i,"(n=%d)"%(len(pmr)))
			t0 = time.time()
			if not initiates is None:
				initiate = initiates[i]
			else:
				initiate = None
			alphas,betas,mu_a,mu_b,tau_a,tau_b,m4_cis,cov,lnj = M4(pmr,maxiter=maxiter,nres=nres,initiate=initiate,stoptoobig=stoptoobig)
			t1 = time.time()
			print('[%.6f, %.6f, %.6f, %.6f], ### %.3f '%(mu_a,mu_b,tau_a,tau_b,lnj))
			print('Time:',t1-t0)
			p = alphas/(alphas+betas)
			alphass.append(alphas)
			betass.append(betas)
			mu_as.append(mu_a)
			mu_bs.append(mu_b)
			tau_as.append(tau_a)
			tau_bs.append(tau_b)
			ps.append(m4_cis[:,1])
			covs.append(cov)
		mu_as = np.array(mu_as)
		mu_bs = np.array(mu_bs)
		tau_as = np.array(tau_as)
		tau_bs = np.array(tau_bs)
		covs = np.array(covs)
		
		if not unique_name == '':
			with h5py.File(unique_name,'w') as f:
				for i in range(len(p_ress)):
					f.create_dataset('ps/%d'%(i),data=ps[i])
				f.create_dataset('mu_as',data=mu_as)
				f.create_dataset('mu_bs',data=mu_bs)
				f.create_dataset('tau_as',data=tau_as)
				f.create_dataset('tau_bs',data=tau_bs)
				f.create_dataset('covs',data=covs)
			print('SAVED',unique_name)

	return ps,mu_as,mu_bs,tau_as,tau_bs,covs

def process_sets_indiv(p_ress,alpha0=1.,beta0=1.):
	'''
	'''
	thetass = []
	ciss = []
	ps = []
	for i in range(len(p_ress)):
		pmr = p_ress[i]
		print('Running',i,"(n=%d)"%(len(pmr)))
		thetas,cis = M2(pmr,alpha0,beta0)
		thetass.append(thetas)
		ciss.append(cis)
		ps.append(cis[:,1])

	return ps
