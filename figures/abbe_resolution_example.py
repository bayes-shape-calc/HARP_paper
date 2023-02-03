## 23

import matplotlib.pyplot as plt
import numpy as np






x = np.linspace(-2.,2.,10000)
s2 = .5**2.
r2 = s2/6.

m1 = -.4
m2 =+.4

y1 = np.exp(-.5/r2*(x-m1)**2.)
y2 = np.exp(-.5/r2*(x-m2)**2.)
y3 = np.exp(-.5/s2*(x-m1)**2.)
y4 = np.exp(-.5/s2*(x-m2)**2.)




offset=-2


# for mm in [m2]:
	# plt.axvline(x=mm,color='k',linewidth=1.,linestyle='--')

plt.plot(x*1.1,x*0,color='k',)
plt.plot(x*1.1,x*0+offset,color='k',)

plt.plot(x,y1+y2,color='tab:purple',linestyle='-',linewidth=2.)
plt.plot(x,y1,color='tab:blue',linestyle='--',linewidth=1.)
plt.plot(x,y2,color='tab:red',linestyle='--',linewidth=1.)

plt.plot(x,offset+y3+y4,color='tab:purple',linestyle='-',linewidth=2.)
plt.plot(x,offset+y3,color='tab:blue',linestyle='--',linewidth=1.)
plt.plot(x,offset+y4,color='tab:red',linestyle='--',linewidth=1.)

plt.plot([m2,m2],[0,1],color='k',linestyle='--',linewidth=1)
plt.plot([m2,m2+np.sqrt(-2.*r2*np.log(.5))],[.5,.5],color='k',linestyle='--',marker='None',linewidth=1)

plt.plot([m2,m2],[0+offset,1+offset],color='k',linestyle='--',linewidth=1)
plt.plot([m2,m2+np.sqrt(-2.*s2*np.log(.5))],[offset+.5,offset+.5],color='k',linestyle='--',marker='None',linewidth=1)

from scipy.optimize import minimize
def fxn(theta,x):
	h,m,s2 = theta
	return h*np.exp(-.5/s2*(x-m)**2.)
def minfxn(theta,x,y):
	if s2 < 0:
		return np.inf
	return np.sqrt(np.sum(np.square(y-fxn(theta,x))))

out = minimize(minfxn,np.array((1.,0.,1.)),args=(x,y3+y4))
print(out)
t2 = out.x[2]
m = 0


h = np.exp(-.5/s2*m1**2) + np.exp(-.5/s2*m2**2.)
plt.plot([m,m],[0+offset,h+offset],color='k',linestyle='-',linewidth=2)
plt.plot([m,.839],[offset+.5*h,offset+.5*h],color='k',linestyle='-',marker='None',linewidth=2)

plt.xticks(())
plt.yticks(())
plt.axis('off')

plt.savefig('figures/rendered/abbe_resolution_example.pdf')
plt.savefig('figures/rendered/abbe_resolution_example.png')
