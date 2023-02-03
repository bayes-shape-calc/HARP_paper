## 23

import common_figures as cf
import numpy as np
import matplotlib.pyplot as plt


deposit_date = cf.load_depositdate('./all_results.hdf5')
software = cf.load_software_recon('./all_results.hdf5')
resolution = cf.load_resolution('./all_results.hdf5')
P_res = cf.load_P_res('./all_results.hdf5')
print('loaded')


### apply year cutoff
ycut = 2000
keep =[False for _ in range(len(deposit_date))]
for i in range(len(deposit_date)):
	y,m,d = deposit_date[i].split('-')
	if (int(y) >= ycut) and resolution[i] < 8.:
		keep[i] = True
deposit_date = [deposit_date[i] for i in range(len(deposit_date)) if keep[i]]
software = [software[i] for i in range(len(software)) if keep[i]]
P_res = [P_res[i] for i in range(len(P_res)) if keep[i]]

## get the top software groups
software = [s.lower() for s in software]
software = np.array(software)
ns = np.unique(software)
counts = np.zeros(ns.size)
for i in range(counts.size):
	counts[i] = np.sum(software == ns[i])

order = np.argsort(counts)
ns = ns[order]
counts = counts[order]
keep = counts > 3
keep[np.nonzero(ns == 'missing entry')] = False
software_types = ns[keep]


## create software sets`
P_res_software = [[] for _ in range(software_types.size)]
for i in range(software.size):
	if (software[i].lower() in software_types):
		software_index = np.nonzero(software[i].lower()==software_types)[0][0]
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
ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_res_software,nres=20)

fig,ax = cf.make_fig_set_ab(x,mu_as,mu_bs,tau_as,tau_bs,covs)

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
fig.savefig('figures/rendered/software_reconstruction_%d.pdf'%(ycut))
fig.savefig('figures/rendered/software_reconstruction_%d.png'%(ycut),dpi=300)
plt.show()
