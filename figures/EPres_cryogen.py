## 23

import common_figures as cf
import numpy as np
import matplotlib.pyplot as plt


deposit_date = cf.load_depositdate('./all_results.hdf5')
cryogen = cf.load_cryogen('./all_results.hdf5')
# humidity = cf.load_humidity('./all_results.hdf5')
resolution = cf.load_resolution('./all_results.hdf5')
P_res = cf.load_P_res('./all_results.hdf5')
print('loaded')


### apply year cutoff
ycut = 2018
rescutoff = 8.0
keep =[False for _ in range(len(deposit_date))]
for i in range(len(deposit_date)):
	y,m,d = deposit_date[i].split('-')
	if (int(y) >= ycut) and resolution[i] < rescutoff:
		keep[i] = True
deposit_date = [deposit_date[i] for i in range(len(deposit_date)) if keep[i]]
cryogen = [cryogen[i] for i in range(len(cryogen)) if keep[i]]
# humidity = [humidity[i] for i in range(len(humidity)) if keep[i]]
P_res = [P_res[i] for i in range(len(P_res)) if keep[i]]

cryogen = [c.lower() for c in cryogen]
cryogen = np.array(cryogen)



ns = np.unique(cryogen)
counts = np.zeros(ns.size)
for i in range(counts.size):
	counts[i] = np.sum(cryogen == ns[i])

order = np.argsort(counts)
ns = ns[order]
counts = counts[order]
keep = counts > 5
keep[np.nonzero(ns == 'missing entry')] = False
cryogen_types = ns[keep]

print(counts[keep],cryogen_types)


## create software sets`
P_res_cryogen = [[] for _ in range(cryogen_types.size)]
for i in range(cryogen.size):
	if (cryogen[i] in cryogen_types):
		cryogen_index = np.nonzero(cryogen[i]==cryogen_types)[0][0]
		pp = P_res[i].copy()
		pp[~np.isfinite(pp)] = 0.
		P_res_cryogen[cryogen_index].append(pp)
print('sorted')



###  cap groups at 3000
for i in range(len(P_res_cryogen)):
	ind = 1
	while len(P_res_cryogen[i]) > 3000:
		if ind == 1:
			cryogen_types[i] = cryogen_types[i]+' (split)'# + ' %d'%(ind)
		ind += 1
		rvss = np.unique(np.random.randint(low=0,high=len(P_res_cryogen[i]),size=3000))
		newp = [P_res_cryogen[i][j] for j in range(len(P_res_cryogen[i])) if j in rvss]
		P_res_cryogen.append(newp)
		cryogen_types = np.append(cryogen_types,cryogen_types[i])#[:-2]+' %d'%(ind))
		P_res_cryogen[i] = [P_res_cryogen[i][j] for j in range(len(P_res_cryogen[i])) if j not in rvss]


# # ps,fig,ax = cf.process_sets_indiv(x,P_res_months)
ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_res_cryogen,'figures/models/hmodel_cryogen.hdf5',nres=5)



order = (np.exp(mu_as)/(np.exp(mu_as)+np.exp(mu_bs))).argsort()
mu_as = mu_as[order]
mu_bs = mu_bs[order]
tau_as = tau_as[order]
tau_bs = tau_bs[order]
covs = covs[order]
cryogen_types = cryogen_types[order]

x = np.arange(cryogen_types.size)

fig,ax = cf.make_fig_set_abtautoo(x,mu_as,mu_bs,tau_as,tau_bs,covs)

for aa in fig.axes:
	aa.set_xlim(x.min()-.1,x.max()+.1)
	aa.set_xticks(x)
	aa.set_xticklabels(cryogen_types,rotation=45,ha='right')
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
fig.savefig('figures/rendered/EPres_cryogen.pdf')
fig.savefig('figures/rendered/EPres_cryogen.png',dpi=300)
plt.show()
