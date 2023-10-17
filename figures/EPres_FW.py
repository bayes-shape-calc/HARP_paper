import common_figures as cf
import numpy as np
import matplotlib.pyplot as plt


resolution = cf.load_resolution('./all_results.hdf5')
formula_weights = cf.load_formula_weight('./all_results.hdf5')
P_res = cf.load_P_res('./all_results.hdf5')
deposit_date = cf.load_depositdate('./all_results.hdf5')
print('loaded')

# ycut = 2018
# keep =[False for _ in range(len(deposit_date))]
# for i in range(len(deposit_date)):
# 	y,m,d = deposit_date[i].split('-')
# 	if (int(y) >= ycut):
# 		keep[i] = True
# keep = np.bitwise_and(keep,resolution<4.0)
keep = resolution < 3.2
keep = np.bitwise_and(np.array(keep),np.isfinite(formula_weights))

resolution = np.array([resolution[i] for i in range(keep.size) if keep[i]])
formula_weights = np.array([formula_weights[i] for i in range(keep.size) if keep[i]])
P_res = [P_res[i] for i in range(keep.size) if keep[i]]



fws = np.logspace(np.log10(formula_weights.min()-1),np.log10(formula_weights.max()+1),100)
P_res_fw = [[] for _ in range(fws.size-1)]
for i in range(formula_weights.size):
	pp = P_res[i].copy()
	pp[~np.isfinite(pp)] = 0.
	index = np.argmax(formula_weights[i] < fws) -1
	P_res_fw[index].append(pp)
print('sorted')

ns = np.array([len(prfi) for prfi in P_res_fw])
keep = ns > 5
fwsk = fws[:-1][keep]
P_res_fwk = [P_res_fw[i] for i in range(len(P_res_fw)) if keep[i]]
x = np.arange(fws.size-1)
x = x[keep]

# mps,fig = cf.process_sets_indiv(resk,P_res_resk,width=.07,alpha0=1.,beta0=1.)
ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_res_fwk,'figures/models/hmodel_fw.hdf5',nres=20)
fig,ax = cf.make_fig_set_abtautoo(fwsk,mu_as,mu_bs,tau_as,tau_bs,covs)

for aa in fig.axes:
	aa.set_xscale('log')
# 	aa.set_xlim(fwsk.min()*.9,fwsk.max()*1.1)
# 	# aa.set_xticks(fws)
# 	# aa.set_xticklabels(fws)

ax['A'].set_ylim([.1,1.])
ax['P'].set_ylim([.4,1.])
ax['A'].yaxis.set_major_formatter(plt.ScalarFormatter())
# ax['B'].yaxis.set_major_formatter(plt.ScalarFormatter())
# ax['A'].yaxis.set_minor_formatter(plt.ScalarFormatter())
# ax['B'].yaxis.set_minor_formatter(plt.ScalarFormatter())
ax['P'].set_xlabel('Formula Weight (Da)')
ax['B'].set_xlabel('Formula Weight (Da)')
ax['T'].set_xlabel('Formula Weight (Da)')
# ax['A'].set_ylim(6e-2,2e1)
#
fig.savefig('figures/rendered/EPres_fw.pdf')
fig.savefig('figures/rendered/EPres_fw.png',dpi=300)
#
plt.close()
