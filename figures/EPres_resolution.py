## 23

import common_figures as cf
import numpy as np
import matplotlib.pyplot as plt


resolution = cf.load_resolution('./all_results.hdf5')
deposit_date = cf.load_depositdate('./all_results.hdf5')
P_res = cf.load_P_res('./all_results.hdf5')
print('loaded')


for ycut in [2000,2014,2018,2019]:
	res = np.arange(0,8.,.1)
	P_res_res = [[] for _ in range(res.size)]
	for i in range(len(deposit_date)):
		y,m,d = deposit_date[i].split('-')
		if int(y) >= ycut:
			pp = P_res[i].copy()
			pp[~np.isfinite(pp)] = 0.
			if resolution[i] > res[-1]:
				index = res.size-1
			else:
				index = np.argmax(res >= resolution[i])-1
			P_res_res[index].append(pp)
	print('sorted')

	ns = np.array([len(prri) for prri in P_res_res])
	keep = ns > 1
	resk = res[keep]
	P_res_resk = [P_res_res[i] for i in range(len(P_res_res)) if keep[i]]

	# mps,fig = cf.process_sets_indiv(resk,P_res_resk,width=.07,alpha0=1.,beta0=1.)
	ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_res_resk,nres=20)
	fig,ax = cf.make_fig_set_ab(resk,mu_as,mu_bs,tau_as,tau_bs,covs)

	for aa in fig.axes:
		aa.set_xticks(np.arange(8+1))
		aa.set_xticklabels(np.arange(8+1))
		aa.set_xlim(0.,8.)
	ax['P'].set_xlabel('Resolution ($\AA$)')
	ax['B'].set_xlabel('Resolution ($\AA$)')
	ax['A'].set_ylim(6e-2,2e1)

	fig.savefig('figures/rendered/EPres_resolution_%d-2022.pdf'%(ycut))
	fig.savefig('figures/rendered/EPres_resolution_%d-2022.png'%(ycut),dpi=300)

	plt.show()
