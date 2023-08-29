## Broken?

## 23

import common_figures as cf
import numpy as np
import matplotlib.pyplot as plt
import h5py

resolution = cf.load_resolution('./all_results.hdf5')
P_res = cf.load_P_res('./all_results.hdf5')
pdbids = cf.load_pdbids('./all_results.hdf5')
chains = cf.load_chains('./all_results.hdf5')
resids = cf.load_resids('./all_results.hdf5')

radial = []
with h5py.File('all_radial.hdf5','r') as f:
	for pdbid in pdbids: ## will be ordered the same this way
		radial.append(f[pdbid]['radial'][:])
print('loaded')

for i in range(len(P_res)):
	P_res[i][np.isnan(P_res[i])] = 0.

# keep = np.ones(len(P_res),dtype='bool')
keep = resolution < 3.2

		

radial = [radial[i] for i in range(keep.size) if keep[i]]
P_res = [P_res[i] for i in range(keep.size) if keep[i]]

coms = [ri[0] for ri in radial]
radial = [ri[1:] for ri in radial]

rmin = np.array([ri.min(0) for ri in radial])
rmax = np.array([ri.max(0) for ri in radial])
cube = np.product(rmax-rmin,axis=1)
ind = cube.argmax()
ind = 1

## want (3,N) for np.cov to work
dr = [radial[i]-coms[i][None,:] for i in range(len(radial))]
covs = np.array([np.cov(dri.T) for dri in dr])
w,v = np.linalg.eig(covs)
w = w.real
v = v.real
#
# mmin = np.array([np.dot(dr[i],v[i].T).min(0) for i in range(len(dr))])
# mmax = np.array([np.dot(dr[i],v[i].T).max(0) for i in range(len(dr))])
# dmm = mmax-mmin
# cube2 = np.product(dmm,1)

ww = np.sqrt(w) * 3.

# from scipy.integrate import quad
# out = quad(lambda x: 4.*np.pi*x**2.* 1./np.sqrt(2.*np.pi*1.*1.)**3.* np.exp(-.5/1./1.*(x-0)**2.),0,3.)
# print(out)

theta = np.arctan(ww.min(1)/ww.max(1))
vol = 4/3.*np.pi*np.product(ww,axis=1)
rapp = (vol/(4/3.*np.pi))**(1./3.)

rapp  = np.array([np.sqrt(np.mean(dri**2.)) for dri in dr])


xorder = rapp.argsort()
ngroup = 100
n = xorder.size//ngroup + 1
P_ress = []
rr = []
for i in range(n):
	if i == 0:
		P = []
		rmin = 0
	else:
		if rmax - rmin  < 1e-10:
			pass
		else:
			P_ress.append(P)
			rr.append((rmax+rmin)*.5)
			rmin = rapp[xorder[i*ngroup]]
			P = []
	for j in range(ngroup):
		ind = i*ngroup+j
		if ind < xorder.size:
			P.append(P_res[xorder[ind]])
			rmax = rapp[xorder[ind]]
rr = np.array(rr)





#
# rr = np.arange(65)*1.25
# rr = .5*(rr[1:]+rr[:-1])
#
# P_ress = []
# for i in range(rr.size-1):
# 	keep = np.bitwise_and(rapp > rr[i],rapp <= rr[i+1])
# 	P_ress.append([P_res[i] for i in range(len(P_res)) if keep[i]])
#
# # rr = .5*(rr[1:]+rr[:-1])
# keep = [len(pr) > 5 for pr in P_ress]
# rr = rr[:-1][keep]
# P_ress = [P_ress[i] for i in range(len(P_ress)) if keep[i]]



# # mps,fig = cf.process_sets_indiv(resk,P_res_resk,width=.07,alpha0=1.,beta0=1.)
ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_ress,'figures/models/hmodel_radial.hdf5',nres=20,maxiter=1000)
# ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_ress,nres=20,maxiter=1000)
fig,ax = cf.make_fig_set_ab(rr,mu_as,mu_bs,tau_as,tau_bs,covs)

lpl = np.exp(mu_as)/(np.exp(mu_as)+np.exp(mu_bs))
ix = np.argmin(np.abs(31.-rr))

ax['P'].annotate(
	"Apoferritin",
	xy = (rr[ix],lpl[ix]),
	xytext = (rr[ix] + 5.,lpl[ix]+.05),
	xycoords='data',
	textcoords='data',
	arrowprops=dict(arrowstyle="->", connectionstyle="arc3")
)


rx = np.array([0,20,40,60,80])
for aa in fig.axes:
	aa.set_xlim(0,rx.max())
	aa.set_xticks(rx)
	aa.set_xticklabels(rx,rotation=0)
	aa.set_xlim(5,65)

ax['P'].set_ylim(.4,.9)
ax['P'].set_xlabel(r'$\langle \vert \delta r \vert \rangle$ ($\AA$)')
ax['B'].set_xlabel(r'$\langle \vert \delta r \vert \rangle$ ($\AA$)')
# ax['T'].set_xlabel(r'$\langle \vert \delta r \vert \rangle$ ($\AA$)')
fig.tight_layout()
fig.subplots_adjust(bottom=.22)

fig.savefig('figures/rendered/Epres_radial.pdf')
fig.savefig('figures/rendered/Epres_radial.png',dpi=300)
plt.close()
