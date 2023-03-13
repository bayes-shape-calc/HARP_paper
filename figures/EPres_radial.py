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

radial = []
with h5py.File('all_radial.hdf5','r') as f:
	for pdbid in f.keys():
		radial.append(f[pdbid]['radial'][:])
print('loaded')


resolution_cutoff = 3.0

rad = np.concatenate([np.linspace(0,200,50+1),[np.inf,]])
P_res_rad = [[] for _ in range(rad.size-1)]
for i in range(len(resolution)):
	if resolution[i] < resolution_cutoff:
		pp = P_res[i].copy()
		pp[~np.isfinite(pp)] = 0.
		
		for ri in range(rad.size-1):
			keep = np.bitwise_and(radial[i] >= rad[ri], radial[i] < rad[ri+1])
			if np.sum(keep)>1:
				P_res_rad[ri].append(pp[np.nonzero(keep)])
print('sorted')


# ### TEST LIMITS
# for i in range(len(P_res_rad)):
# 	if len(P_res_rad[i]) > 100:
# 		P_res_rad[i] = P_res_rad[i][:100]



ns = np.array([len(prri) for prri in P_res_rad])
keep = ns > 1
radk = rad[:-1][keep]
P_res_radk = [P_res_rad[i] for i in range(len(P_res_rad)) if keep[i]]





# # mps,fig = cf.process_sets_indiv(resk,P_res_resk,width=.07,alpha0=1.,beta0=1.)
# ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_res_radk,'figures/models/hmodel_radial_%.1fA.hdf5'%(resolution_cutoff),nres=20,maxiter=1000)

fig,ax = cf.make_fig_set_p(radk+(rad[1]-rad[0])/2.,P_res_radk,yscale='logit',width=.7)


rx = [0,25,50,75,100,125,150,175,200]
for aa in fig.axes:
	aa.set_xlim(radk.min(),radk.max())
	aa.set_xticks(rx)
	aa.set_xticklabels(rx,rotation=-45)

# ax['P'].set_xlabel('Radial distance ($\AA$)')
# ax['B'].set_xlabel('Radial distance ($\AA$)')
fig.savefig('figures/rendered/Epres_radial_%.1fA.pdf'%(resolution_cutoff))
fig.savefig('figures/rendered/Epres_radial_%.1fA.png'%(resolution_cutoff),dpi=300)
plt.show()
