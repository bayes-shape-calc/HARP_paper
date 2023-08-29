## 23

import common_figures as cf
import numpy as np
import matplotlib.pyplot as plt


deposit_date = cf.load_depositdate('./all_results.hdf5')
software,version = cf.load_software_recon('./all_results.hdf5')
resolution = cf.load_resolution('./all_results.hdf5')
P_res = cf.load_P_res('./all_results.hdf5')
print('loaded')


### apply year cutoff
ycut = 2018
keep =[False for _ in range(len(deposit_date))]
for i in range(len(deposit_date)):
	y,m,d = deposit_date[i].split('-')
	if (int(y) >= ycut) and resolution[i] < 8.:
		keep[i] = True
deposit_date = [deposit_date[i] for i in range(len(deposit_date)) if keep[i]]
software = [software[i] for i in range(len(software)) if keep[i]]
version = [version[i] for i in range(len(version)) if keep[i]]
P_res = [P_res[i] for i in range(len(P_res)) if keep[i]]

## get the top software groups
software = [s.lower() for s in software]
software = np.array(software)
version = np.array(version)



keep = np.bitwise_and(software == 'missing entry',np.array(([vi.lower().startswith('relion 2.0') for vi in version])))
software[keep] = 'relion 2'
keep = np.bitwise_and(software == 'missing entry',np.array(([vi.lower().startswith('relion') for vi in version])))
software[keep] = 'relion ?'
keep = np.bitwise_and(software == 'missing entry',np.array(([vi.lower().startswith('cryosparc') for vi in version])))
software[keep] = 'cryosparc ?'


for v in ['0.','v.0','v0.']:
	keep = np.bitwise_and(software == 'cryosparc',np.array(([vi.startswith(v) for vi in version])))
	software[keep] = 'cryosparc 0'
for v in ['1','V1']:
	keep = np.bitwise_and(software == 'cryosparc',np.array(([vi.startswith(v) for vi in version])))
	software[keep] = 'cryosparc 1'
for v in ['2','V2','v2','v.2']:
	keep = np.bitwise_and(software == 'cryosparc',np.array(([vi.startswith(v) for vi in version])))
	software[keep] = 'cryosparc 2'
software[np.bitwise_and(software == 'cryospace',version == 'version 2')] = 'cryosparc 2'
for v in ['3','V3','v3','v.3']:
	keep = np.bitwise_and(software == 'cryosparc',np.array(([vi.startswith(v) for vi in version])))
	software[keep] = 'cryosparc 3'


keep = np.bitwise_and(software == 'relion',np.array(([vi.startswith('1.') for vi in version])))
software[keep] = 'relion 1'
keep = np.bitwise_and(software == 'relion',np.array(([vi.startswith('2') for vi in version])))
software[keep] = 'relion 2'
keep = np.bitwise_and(software == 'relion',np.array(([vi.startswith('3') for vi in version])))
software[keep] = 'relion 3'
keep = np.bitwise_and(software == 'relion',version=='RELION 3.1')
software[keep] = 'relion 3'
keep = np.bitwise_and(software == 'relion',version=='Relion3.1')
software[keep] = 'relion 3'
keep = np.bitwise_and(software == 'relion',np.array(([vi.startswith('4') for vi in version])))
software[keep] = 'relion 4'


for soft in np.unique(software):
	print(soft,np.unique(version[np.nonzero(software == soft)]))


ns = np.unique(software)
counts = np.zeros(ns.size)
for i in range(counts.size):
	counts[i] = np.sum(software == ns[i])

order = np.argsort(counts)
ns = ns[order]
counts = counts[order]
keep = counts > 5
keep[np.nonzero(ns == 'missing entry')] = False
software_types = ns[keep]


## create software sets`
P_res_software = [[] for _ in range(software_types.size)]
for i in range(software.size):
	if (software[i] in software_types):
		software_index = np.nonzero(software[i]==software_types)[0][0]
		pp = P_res[i].copy()
		pp[~np.isfinite(pp)] = 0.
		P_res_software[software_index].append(pp)
print('sorted')

#
# ### testing -- cap at 3000
# for i in range(len(P_res_software)):
# 	if len(P_res_software[i]) > 3000:
# 		P_res_software[i] = [P_res_software[i][rvs] for rvs in np.random.randint(low=0,high=len(P_res_software),size=3000)]




# # ps,fig,ax = cf.process_sets_indiv(x,P_res_months)
ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_res_software,'figures/models/hmodel_software.hdf5',nres=5)



order = (np.exp(mu_as)/(np.exp(mu_as)+np.exp(mu_bs))).argsort()
mu_as = mu_as[order]
mu_bs = mu_bs[order]
tau_as = tau_as[order]
tau_bs = tau_bs[order]
covs = covs[order]
software_types = software_types[order]

keep = [s not in ['coot','phenix','frealign','frealix'] for s in software_types]
mu_as = mu_as[keep]
mu_bs = mu_bs[keep]
tau_as = tau_as[keep]
tau_bs = tau_bs[keep]
covs = covs[keep]
software_types = software_types[keep]
x = np.arange(len(software_types))

fig,ax = cf.make_fig_set_abtautoo(x,mu_as,mu_bs,tau_as,tau_bs,covs)

for aa in fig.axes:
	aa.set_xlim(x.min()-.1,x.max()+.1)
	aa.set_xticks(x)
	aa.set_xticklabels(software_types,rotation=45,ha='right')
# ax['P'].set_xlabel('Month')
# ax['B'].set_xlabel('Month')
# ax['P'].set_ylim(.1,.3)
# ax['P'].set_yticks([.1,.15,.2,.25,.3])
#
# ax['A'].yaxis.set_major_formatter(plt.ScalarFormatter())
# ax['B'].yaxis.set_major_formatter(plt.ScalarFormatter())
# ax['A'].yaxis.set_minor_formatter(plt.ScalarFormatter())
# ax['B'].yaxis.set_minor_formatter(plt.ScalarFormatter())
#
fig.subplots_adjust(bottom=.22)
#
fig.savefig('figures/rendered/software_reconstruction.pdf')
fig.savefig('figures/rendered/software_reconstruction.png',dpi=300)
plt.close()
