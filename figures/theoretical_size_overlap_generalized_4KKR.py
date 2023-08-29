import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize


def fxn1(r,s):
	## 3D isotropic gaussian
	return 4.*np.pi*(2*np.pi*s*s)**(-1.5)*np.exp(-.5/s/s*r*r) * r*r
def minfxn1(theta,s,density_cutoff):
	## R is the radial distance at which the encompassed density (of a 3D gaussian with sigma=s) is the value cutoff
	R = theta[0]
	if R < 1e-6:
		return np.inf
	integral = quad(fxn1,0.,R,args=(s,))[0]
	return np.sqrt((density_cutoff-integral)**2.)
def calc_radial_cutoff(sigma,cutoff):
	## calculates the distance at which cutoff amount of density is encompassed by an isotropic 3D Gaussian with sigma^2 variance
	guess = np.array((1.,))
	out = minimize(minfxn1,guess,args=(sigma,cutoff),method='Nelder-Mead')
	return out.x[0]


def minfxn2(theta,distance_target,density_fraction):
	### distance_target is the distance between atoms (i.e., C-C = 1.54 A). Density fraction is the encompassed amount of density at that distance. Solving for the sigma of the isotropic 3D gaussian
	sigma = theta[0]
	if sigma <= 1e-6:
		return np.inf
	r = calc_radial_cutoff(sigma,density_fraction)
	return np.sqrt((distance_target-r)**2.)
def find_sigma_resolution_criterion(distance_target,density_fraction):
	## optimize the sigma of an isotropic 3D gaussian that yields density_fraction amount of encompassed density at distance_target length away from the center.
	guess = np.array((1.,))
	out = minimize(minfxn2,guess,args=(distance_target,density_fraction,),method='Nelder-Mead')
	return out.x[0]


fraction = 0.5
for target_bondlength in [1.54,1.536,4.31,10.5]:
	sigma_opt = find_sigma_resolution_criterion(target_bondlength,fraction)
	midpoint_value = np.exp(-.5/(sigma_opt**2.)*(target_bondlength/2)**2.)

	print('Optimized sigma (%.6f A, %f %%):  %.8f A'%(target_bondlength,fraction,sigma_opt))
	print('Midpoint PDF height:     %.6f'%(midpoint_value))
	print('Rayleigh midpoint value: %.3f (see Sheppard 2017)'%(0.735))
	print('\n')
