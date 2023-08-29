## 23

import common_figures as cf
import numpy as np
import matplotlib.pyplot as plt


deposit_date = cf.load_depositdate('./all_results.hdf5')
journal = cf.load_journal('./all_results.hdf5')
resolution = cf.load_resolution('./all_results.hdf5')
P_res = cf.load_P_res('./all_results.hdf5')
print('loaded')


### apply year cutoff
ycut = 2018
rescut = 8.
keep =[False for _ in range(len(deposit_date))]
for i in range(len(deposit_date)):
	y,m,d = deposit_date[i].split('-')
	if (int(y) >= ycut) and resolution[i] < rescut:
		keep[i] = True
deposit_date = [deposit_date[i] for i in range(len(deposit_date)) if keep[i]]
journal = [journal[i] for i in range(len(journal)) if keep[i]]
P_res = [P_res[i] for i in range(len(P_res)) if keep[i]]

# ## get the top software groups
# software = [s.lower() for s in software]
# software = np.array(software)
# ns = np.unique(software)
# counts = np.zeros(ns.size)
# for i in range(counts.size):
# 	counts[i] = np.sum(software == ns[i])
#
# order = np.argsort(counts)
# ns = ns[order]
# counts = counts[order]
# keep = counts > 20
# keep[np.nonzero(ns == 'missing entry')] = False
# software_types = ns[keep]


## create software sets`
journal_types = np.unique(journal)
journal_types = journal_types[journal_types!='other']
journal_types = journal_types[journal_types!='To be pub.']
P_res_journal = [[] for _ in range(journal_types.size)]
for i in range(len(journal)):
	if (journal[i] in journal_types):
		journal_index = np.nonzero(journal[i]==journal_types)[0][0]
		pp = P_res[i].copy()
		pp[~np.isfinite(pp)] = 0.
		P_res_journal[journal_index].append(pp)
print('sorted')



# ### testing -- cap at 500
# for i in range(len(P_res_software)):
# 	if len(P_res_software[i]) > 500:
# 		P_res_software[i] = [P_res_software[i][rvs] for rvs in np.random.randint(low=0,high=len(P_res_software),size=500)]



x = np.arange(len(journal_types))
# # ps,fig,ax = cf.process_sets_indiv(x,P_res_months)
ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_res_journal,'figures/models/hmodel_journal.hdf5',nres=5)



order = (np.exp(mu_as)/(np.exp(mu_as)+np.exp(mu_bs))).argsort()
mu_as = mu_as[order]
mu_bs = mu_bs[order]
tau_as = tau_as[order]
tau_bs = tau_bs[order]
covs = covs[order]
journal_types = journal_types[order]



fig,ax = cf.make_fig_set_abtautoo(x,mu_as,mu_bs,tau_as,tau_bs,covs)

for aa in fig.axes:
	aa.set_xlim(x.min()-.1,x.max()+.1)
	aa.set_xticks(x)
	aa.set_xticklabels(journal_types,rotation=45,ha='right')
# ax['P'].set_xlabel('Month')
# ax['B'].set_xlabel('Month')

ax['P'].set_ylim(.35,.55)
ax['P'].set_yticks([.35,.4,.45,.5,.55])
# ax['P'].set_yticks([.1,.15,.2,.25,.3])
#
ax['A'].yaxis.set_major_formatter(plt.ScalarFormatter())
ax['B'].yaxis.set_major_formatter(plt.ScalarFormatter())
ax['A'].yaxis.set_minor_formatter(plt.ScalarFormatter())
ax['B'].yaxis.set_minor_formatter(plt.ScalarFormatter())
#
fig.subplots_adjust(bottom=.22)
#
fig.savefig('figures/rendered/EPres_journal.pdf')
fig.savefig('figures/rendered/EPres_journal.png',dpi=300)
plt.close()
