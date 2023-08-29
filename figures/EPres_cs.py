## 23

import common_figures as cf
import numpy as np
import matplotlib.pyplot as plt


deposit_date = cf.load_depositdate('./all_results.hdf5')
resolution = cf.load_resolution('./all_results.hdf5')
P_res = cf.load_P_res('./all_results.hdf5')


cs = cf.load_cs('./all_results.hdf5')
num_particles = cf.load_numparticles('./all_results.hdf5')
num_particles_selected = cf.load_numparticles_selected('./all_results.hdf5')
num_imgs = cf.load_numimgs('./all_results.hdf5')


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
P_res = [P_res[i] for i in range(len(P_res)) if keep[i]]
num_particles = num_particles[keep]
num_particles_selected = num_particles_selected[keep]
num_img = num_imgs[keep]
cs = cs[keep]


for csi,count in zip(np.unique(cs),[np.sum(cs==nsi) for nsi in np.unique(cs)]):
	print(csi,count)
### NOTES
## REAALLLLLY many different CS values.... not worth it...




#
# ns = np.unique(humidity)
# counts = np.zeros(ns.size)
# for i in range(counts.size):
# 	counts[i] = np.sum(humidity == ns[i])
#
# order = np.argsort(counts)
# ns = ns[order]
# counts = counts[order]
# keep = counts > 5
# keep[np.nonzero(ns == 'missing entry')] = False
# humidity_types = ns[keep]
#
# print(counts[keep])
# print(humidity_types)
#
#
# ## create software sets`
# P_res_humidity = [[] for _ in range(humidity_types.size)]
# for i in range(humidity.size):
# 	if (humidity[i] in humidity_types):
# 		humidity_index = np.nonzero(humidity[i]==humidity_types)[0][0]
# 		pp = P_res[i].copy()
# 		pp[~np.isfinite(pp)] = 0.
# 		P_res_humidity[humidity_index].append(pp)
# print('sorted')
#
#
# #
# # ###  cap groups at 3000
# # for i in range(len(P_res_humidity)):
# # 	ind = 1
# # 	while len(P_res_humidity[i]) > 3000:
# # 		if ind == 1:
# # 			humidity_types[i] = humidity_types[i]+' (split)'# + ' %d'%(ind)
# # 		ind += 1
# # 		rvss = np.unique(np.random.randint(low=0,high=len(P_res_humidity[i]),size=3000))
# # 		newp = [P_res_humidity[i][j] for j in range(len(P_res_humidity[i])) if j in rvss]
# # 		P_res_humidity.append(newp)
# # 		humidity_types = np.append(humidity_types,humidity_types[i])#[:-2]+' %d'%(ind))
# # 		P_res_humidity[i] = [P_res_humidity[i][j] for j in range(len(P_res_humidity[i])) if j not in rvss]
# #
#
# # # ps,fig,ax = cf.process_sets_indiv(x,P_res_months)
# ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_res_humidity,'figures/models/hmodel_humidity.hdf5',nres=5)
#
#
#
# order = np.array(humidity_types.argsort())
# mu_as = mu_as[order]
# mu_bs = mu_bs[order]
# tau_as = tau_as[order]
# tau_bs = tau_bs[order]
# covs = covs[order]
# humidity_types = humidity_types[order]
#
# x = humidity_types.copy()#np.arange(humidity_types.size)
#
# fig,ax = cf.make_fig_set_abtautoo(x,mu_as,mu_bs,tau_as,tau_bs,covs)
#
# for aa in fig.axes:
# 	aa.set_xlim(x.min()-.1,x.max()+.1)
# 	aa.set_xticks(x)
# 	aa.set_xticklabels(humidity_types,rotation=45,ha='right')
# # ax['P'].set_xlabel('Month')
# # ax['B'].set_xlabel('Month')
# # ax['P'].set_ylim(.1,.3)
# # ax['P'].set_yticks([.1,.15,.2,.25,.3])
# #
# # ax['A'].yaxis.set_major_formatter(plt.ScalarFormatter())
# # ax['B'].yaxis.set_major_formatter(plt.ScalarFormatter())
# # ax['A'].yaxis.set_minor_formatter(plt.ScalarFormatter())
# # ax['B'].yaxis.set_minor_formatter(plt.ScalarFormatter())
# #
# fig.subplots_adjust(bottom=.22)
# #
# fig.savefig('figures/rendered/EPres_humidity.pdf')
# fig.savefig('figures/rendered/EPres_humidity.png',dpi=300)
# plt.show()
