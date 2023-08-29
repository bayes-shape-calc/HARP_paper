## 23

import common_figures as cf
import numpy as np
import matplotlib.pyplot as plt

pdbids = cf.load_pdbids('./all_results.hdf5')
deposit_date = cf.load_depositdate('./all_results.hdf5')
software,version = cf.load_software_model('./all_results.hdf5')
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
pdbids = [pdbids[i] for i in range(len(pdbids)) if keep[i]]

## get the top software groups
software = [s.lower() for s in software]
software = np.array(software)
version = np.array(version)
pdbids = np.array(pdbids)
	
#
# keep = np.bitwise_and(software == 'isolde',np.array(([vi.startswith('1.1') for vi in version])))
# software[keep] = 'isolde'
# keep = np.bitwise_and(software == 'isolde',np.array(([vi.startswith('1.2') for vi in version])))
# software[keep] = 'isolde'
# keep = np.bitwise_and(software == 'isolde',np.array(([vi.startswith('1.3') for vi in version])))
# software[keep] = 'isolde'
# keep = np.bitwise_and(software == 'isolde',np.array(([vi.startswith('1') for vi in version]))) ## keep this last
# software[keep] = 'isolde'
# keep = software == 'isolde'
# software[keep] = 'isolde'
#
# keep = np.bitwise_and(software == 'coot',np.array(([vi.startswith('0.8') for vi in version])))
# software[keep] = 'coot'
# keep = np.bitwise_and(software == 'coot',np.array(([vi.startswith('0.9') for vi in version])))
# software[keep] = 'coot'
# keep = np.bitwise_and(software == 'coot',np.array(([vi.count('9.')>0 for vi in version])))
# software[keep] = 'coot'
# keep = software == 'coot'
# software[keep] = 'coot'
#
# keep = np.bitwise_and(software == 'cryosparc',np.array(([vi.startswith('2.') for vi in version])))
# software[keep] = 'cryosparc'
# keep = np.bitwise_and(software == 'cryosparc',np.array(([vi.startswith('3.') for vi in version])))
# software[keep] = 'cryosparc'
# keep = software == 'cryosparc'
# software[keep] = 'cryosparc'
#
# keep = np.bitwise_and(software == 'rosetta',np.array(([vi.startswith('2018.') for vi in version])))
# software[keep] = 'rosetta 20XX'
# keep = np.bitwise_and(software == 'rosetta',np.array(([vi.startswith('2019') for vi in version])))
# software[keep] = 'rosetta 20XX'
# keep = np.bitwise_and(software == 'rosetta',np.array(([vi.startswith('2020') for vi in version])))
# software[keep] = 'rosetta 20XX'
# keep = np.bitwise_and(software == 'rosetta',np.array(([vi.startswith('3') for vi in version])))
# software[keep] = 'rosetta 3'
# keep = software == 'rosetta'
# software[keep] = 'rosetta ?'

# keep = software == 'refmac'
# software[keep] = 'refmac 5'

#
# keep = np.bitwise_and(software == 'phenix',np.array(([vi.startswith('dev') for vi in version])))
# software[keep] = 'phenix ?'
# for v in ['1.11','1.12','1.13','1.14','1.15','1.16','1.17','1.18','1.19','1.20']:
# 	keep = np.bitwise_and(software == 'phenix',np.array(([vi.startswith(v) for vi in version])))
# 	software[keep] = 'phenix>1.10'
# for v in ['0.9','1.0','1.1','1.2','1.3','1.4','1.5','1.6','1.7','1.8','1.9','1.10']:
# 	keep = np.bitwise_and(software == 'phenix',np.array(([vi.startswith(v) for vi in version])))
# 	software[keep] = 'phenix<=1.10'
# keep = software == 'phenix'
# software[keep] = 'phenix ?'
# keep = np.bitwise_and(software == 'missing entry',np.array(([vi.lower().startswith('phenix') for vi in version])))
# software[keep] = 'phenix ?'


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
# ### testing -- cap at 500
# for i in range(len(P_res_software)):
# 	if len(P_res_software[i]) > 500:
# 		P_res_software[i] = [P_res_software[i][rvs] for rvs in np.random.randint(low=0,high=len(P_res_software),size=500)]
#


x = np.arange(len(software_types))
# # ps,fig,ax = cf.process_sets_indiv(x,P_res_months)
ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_res_software,'figures/models/hmodel_software_modelrefinement.hdf5',nres=5)



order = (np.exp(mu_as)/(np.exp(mu_as)+np.exp(mu_bs))).argsort()
mu_as = mu_as[order]
mu_bs = mu_bs[order]
tau_as = tau_as[order]
tau_bs = tau_bs[order]
covs = covs[order]
software_types = software_types[order]


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
fig.savefig('figures/rendered/software_modelrefinement.pdf')
fig.savefig('figures/rendered/software_modelrefinement.png',dpi=300)
plt.show()
