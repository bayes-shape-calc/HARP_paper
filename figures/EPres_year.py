## 23

import common_figures as cf
import numpy as np
import matplotlib.pyplot as plt


deposit_date = cf.load_depositdate('./all_results.hdf5')
P_res = cf.load_P_res('./all_results.hdf5')
print('loaded')

years = np.arange(2008,2023)
P_res_years = [[] for _ in range(years.size)]
for i in range(len(deposit_date)):
	y,m,d = deposit_date[i].split('-')
	year_index = int(y)-years[0]
	if int(y) in years:
		pp = P_res[i].copy()
		pp[~np.isfinite(pp)] = 0.
		P_res_years[year_index].append(pp)
print('sorted')

# ps,fig,ax = cf.process_sets_indiv(years,P_res_years)
ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_res_years,'figures/models/hmodel_year.hdf5',nres=20)


# fig,ax = cf.make_fig_set_ab(years,mu_as,mu_bs,tau_as,tau_bs,covs)
fig,ax = cf.make_fig_set_abtautoo(years,mu_as,mu_bs,tau_as,tau_bs,covs)

for aa in fig.axes:
	aa.set_xticks(years[::2])
	aa.set_xticklabels(years[::2],rotation=0)
	aa.set_xlim(years.min(),years.max())
ax['P'].set_xlabel('Year')
ax['B'].set_xlabel('Year')
ax['P'].set_ylim(-.00,.5)

for axl in ['A','B','T','R']:
	ax[axl].set_yscale('log')

ax['B'].set_ylim(.1,10.)
ax['R'].set_ylim(1.,40.)
ax['T'].set_ylim(.025,2.5)

# ax['A'].set_ylim(.095,.145)

aticks = np.array((.1,.11,.12,.13,.14))
for _ in range(2): ## idk it doesn't always take the first time
	ax['A'].set_yticks(aticks)
	ax['A'].set_yticklabels(['%.2f'%(t) for t in aticks])

fig.savefig('figures/rendered/EPres_year.pdf')
fig.savefig('figures/rendered/EPres_year.png',dpi=300)

plt.show()
