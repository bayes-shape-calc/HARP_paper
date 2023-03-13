## 23

import common_figures as cf
import numpy as np
import matplotlib.pyplot as plt


deposit_date = cf.load_depositdate('./all_results.hdf5')
P_res = cf.load_P_res('./all_results.hdf5')
print('loaded')

for ycut in [2000,2014,2018,2019]:
	months = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
	P_res_months = [[] for _ in range(months.size)]
	for i in range(len(deposit_date)):
		y,m,d = deposit_date[i].split('-')
		if int(y) >= ycut:
			month_index = int(m)-1
			pp = P_res[i].copy()
			pp[~np.isfinite(pp)] = 0.
			P_res_months[month_index].append(pp)
	print('sorted')


	x = np.arange(len(months))
	# ps,fig,ax = cf.process_sets_indiv(x,P_res_months)
	ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_res_months,'figures/models/hmodel_month_%d.hdf5'%(ycut),nres=20)

	# fig,ax = cf.make_fig_set_ab(x,mu_as,mu_bs,tau_as,tau_bs,covs)
	fig,ax = cf.make_fig_set_abtautoo(x,mu_as,mu_bs,tau_as,tau_bs,covs)

	for aa in fig.axes:
		# aa.set_xlim(x.min()-.1,x.max()+.1)
		aa.set_xlim(x.min(),x.max())
		aa.set_xticks(x)
		aa.set_xticklabels(months,rotation=45,ha='right')
	ax['P'].set_xlabel('Month')
	ax['B'].set_xlabel('Month')
	ax['T'].set_xlabel('Month')
	ax['P'].set_ylim(.3,.5)
	ax['P'].set_yticks([.3,.35,.4,.45,.5])
	# ax['P'].set_yticks([.1,.15,.2,.25,.3])

	for axl in ['A','B','T','R']:
		ax[axl].yaxis.set_major_formatter(plt.ScalarFormatter())
		ax[axl].yaxis.set_major_formatter(plt.ScalarFormatter())
	# ax['A'].yaxis.set_minor_formatter(plt.ScalarFormatter())
	# ax['B'].yaxis.set_minor_formatter(plt.ScalarFormatter())

	fig.subplots_adjust(left=.1,wspace=.24,bottom=.17,right=.975,hspace=.055,top=.975)

	fig.savefig('figures/rendered/EPres_months_%d-2022.pdf'%(ycut))
	fig.savefig('figures/rendered/EPres_months_%d-2022.png'%(ycut),dpi=300)
	plt.show()
