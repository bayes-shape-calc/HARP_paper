## 23

##  L.-M. Peng, Electron atomic scattering factors and scattering potentials of crystals. Micron 30, 625–648 (1999).
##  H. Rullgård, L.-G. Öfverstedt, S. Masich, B. Daneholt, O. Öktem, Simulation of transmission electron microscope images of biological specimens: SIMULATION OF TEM IMAGES OF BIOLOGICAL SPECIMENS. Journal of Microscopy 243, 234–256 (2011).


## H. X. Gao, L.-M. Peng, Parameterization of the temperature dependence of the Debye–Waller factors. Acta Crystallogr A Found Crystallogr 55, 926–932 (1999).

import numpy as np
import matplotlib.pyplot as plt


rmax=5.
r = np.linspace(-1,1.,1000)*rmax


carbon_a = np.array((.1361,.5482,1.2266,.5971))
carbon_b = np.array((.3731,3.2814,13.0456,41.0202))

oxygen_a = np.array((.1433, .5103, .9370, .3923))
oxygen_b = np.array((.3055, 2.2683, 8.2625, 25.6645))

nitrogen_a = np.array((.1372, .5344, 1.0862, .4547))
nitrogen_b = np.array((.3287, 2.6733, 10.3165, 32.7631))

phosphorous_a = np.array((.3540, .9397, 2.6203, 1.5707))
phosphorous_b = np.array((.3941, 3.1810, 15.6579, 49.5239))

hydrogen_a = np.array((.0367, .1269, .2360, .1290))
hydrogen_b = np.array((.5608, 3.7913, 13.5557, 37.7229))

sulfur_a = np.array((.3478, .9158, 2.5066, 1.3884))
sulfur_b = np.array((.3652, 2.8915, 13.0522, 40.1848))

params = [
	['Hydrogen',hydrogen_a,hydrogen_b],
	['Carbon',carbon_a,carbon_b],
	['Nitrogen',nitrogen_a,nitrogen_b],
	['Oxygen',oxygen_a,oxygen_b],
	['Phosphorous',phosphorous_a,phosphorous_b],
	['Sulfur',sulfur_a,sulfur_b],
]


def render_theory(theta,r,atom_a,atom_b):
	scale,abs_a,abs_b,B = theta

	# B = 8.*np.pi**2.*dw_a**2.  ## Debye-Waller

	V = np.sum(atom_a[:,None]*(atom_b[:,None]+B)**(-3./2)*np.exp(-4.*np.pi**2.*(r[None,:])**2./(atom_b[:,None]+B)),axis=0)
	# ## it's basically doing this....
	# # v_B = B/(8.*np.pi**2.)
	# # v_bi = atom_b/(8.*np.pi**2.)
	# # V = np.sum(atom_a[:,None]*atom_b[:,None]**(-3./2)*np.exp(-.5*(r**2.)[None,:]/(v_bi[:,None]+v_B)),axis=0)
	#
	# Vabs = abs_a*np.exp(-.5/(abs_b**2.)*np.abs(r)**2.) ## absorption potential`


	VV = V**2. #+ Vabs**2. ### (it's the imaginary component (a+ib) so multiply by complex conjugate (a-ib))
	# V *= 2132.79**2.
	VV *= scale

	return VV


## from Tilton, Dewan, Petsko Biochemistry 1992, 31, 2469. See fig. 7 T = 0,100.
## Also, Carugo amino acids 2018 "atomic displacement parameteres in structural biology" says 6.1 A2, or 5A2 at low temp for trypsinogen and met-myoglobin. see Singh 1980, Hartmann 1982. Hartmann is a Fraunfelder paper -- very good! says it's 5 at cryo temp so use that.
for cutoff,approx_B in zip([.95,.5],[37,143.8]):#,5.,0.]:
	dw_a = np.sqrt(approx_B/2.)/(2.*np.pi)
	print(approx_B,dw_a)


	def fxn1(theta,x):
		a,b = theta
		return a*np.exp(-.5*x**2./(b**2.))
	def minfxn1(theta,x,y):
		a,b = theta
		if a<0 or b < 0:
			return np.inf
		yy = fxn1(theta,x)
		out = np.sqrt(np.sum(np.square(yy-y)))
		return out
	from scipy.optimize import minimize

	fig,ax =plt.subplots(1)

	ymax = 0
	for atom,atom_a,atom_b in params:
		y = render_theory([1.,0.,.3,approx_B],r,atom_a,atom_b)
		ymax = np.max((y.max(),ymax))

	ss = []
	ws = []
	for atom,atom_a,atom_b in [params[1],]:
		y = render_theory([1./ymax,0.,.3,approx_B],r,atom_a,atom_b)
		guess = np.array((1.,.2))
		out = minimize(minfxn1,guess,args=(r,y),method='Nelder-Mead')
		ws.append(out.x[0])
		ss.append(out.x[1])

	print(ws)
	print([t[0] for t in params])

	# cutoff = 0.5
	s = np.mean(ss)

	from scipy.integrate import quad
	def fxn2(r,s):
		return 4.*np.pi*(2*np.pi*s*s)**(-1.5)*np.exp(-.5/s/s*r*r) * r*r
	def minfxn2(theta,s,cutoff):
		R = theta[0]
		if R < 1e-6:
			return np.inf
		integral = quad(fxn2,0.,R,args=(s,))[0]
		return np.abs(cutoff-integral)
	guess = np.array((.2,))
	out = minimize(minfxn2,guess,args=(s,cutoff),method='Nelder-Mead')
	cc = out.x[0]
	
	print(cc)
	ccbond = 1.54
	ax.plot(r-ccbond/2.,fxn1([1.,s],r),color='tab:blue')
	ax.plot(r+ccbond/2.,fxn1([1.,s],r),color='tab:orange')
	

	for xx in [-cc*3/2.,cc/2.]:
		ax.axvline(x=xx,color='tab:blue',lw=1.,linestyle='--')
	for xx in [-cc/2.,cc*3/2.]:
		ax.axvline(x=xx,color='tab:orange',lw=1.,linestyle='--')



	ax.set_title(r'B = %.1f $\AA^2$,  $\langle\sigma\rangle=%.02f \AA$,  $r_{%.0f\%%}= %.2f \AA$'%(approx_B,s,100*cutoff,cc))

	ax.set_xlim(-3.,3.)
	ax.set_xlabel('Distance (Angstrom)')
	# ax[0].legend()
	ax.set_ylabel('Relative EM Radial Density')
	# ax[1].set_ylabel('Residual')
	# ax[1].set_ylim(-.02,.02)
	plt.tight_layout()
	#
	#
	plt.savefig('figures/rendered/fig_theoretical_overlap_B%.0f.pdf'%(approx_B))
	plt.savefig('figures/rendered/fig_theoretical_overlap_B%.0f.png'%(approx_B),dpi=300)
	plt.show()
