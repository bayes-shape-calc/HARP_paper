## 23

#### might have to re-run a while to get the first set to not crash

import common_figures as cf
import numpy as np
import matplotlib.pyplot as plt


deposit_date = cf.load_depositdate('./all_results.hdf5')
camera = cf.load_camera('./all_results.hdf5')
resolution = cf.load_resolution('./all_results.hdf5')
P_res = cf.load_P_res('./all_results.hdf5')
print('loaded')


### apply year cutoff
ycut = 1900
rescut = 8.0
keep =[False for _ in range(len(deposit_date))]
for i in range(len(deposit_date)):
	y,m,d = deposit_date[i].split('-')
	if (int(y) >= ycut) and resolution[i] < rescut:
		keep[i] = True
deposit_date = [deposit_date[i] for i in range(len(deposit_date)) if keep[i]]
camera = [camera[i] for i in range(len(camera)) if keep[i]]
P_res = [P_res[i] for i in range(len(P_res)) if keep[i]]

## get the top camera groups
## create software sets`
camera = [s for s in camera]
camera = np.array(camera)
ns = np.unique(camera)
counts = np.zeros(ns.size)
for i in range(counts.size):
	counts[i] = np.sum(camera == ns[i])

order = np.argsort(counts)
ns = ns[order]
counts = counts[order]
keep = counts > 5
camera_types = ns[keep]

# camera_types = np.unique(camera)


### extract info
camera_types = camera_types[camera_types!='OTHER']
# camera_types = camera_types[camera_types!='To be pub.']
P_res_camera = [[] for _ in range(camera_types.size)]
for i in range(len(camera)):
	if (camera[i] in camera_types):
		camera_index = np.nonzero(camera[i]==camera_types)[0][0]
		pp = P_res[i].copy()
		pp[~np.isfinite(pp)] = 0.
		P_res_camera[camera_index].append(pp)
print('sorted')



### testing -- cap at 2000
for i in range(len(P_res_camera)):
	if len(P_res_camera[i]) > 2000:
		P_res_camera[i] = [P_res_camera[i][rvs] for rvs in np.random.randint(low=0,high=len(P_res_camera),size=2000)]



x = np.arange(len(camera_types))
# # ps,fig,ax = cf.process_sets_indiv(x,P_res_months)
ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_res_camera,'figures/models/hmodel_camera.hdf5',nres=5)

fig,ax = cf.make_fig_set_abtautoo(x,mu_as,mu_bs,tau_as,tau_bs,covs)

for aa in fig.axes:
	aa.set_xlim(x.min()-.1,x.max()+.1)
	aa.set_xticks(x)
	aa.set_xticklabels(camera_types,rotation=45,ha='right')
# ax['P'].set_xlabel('Month')
# ax['B'].set_xlabel('Month')
# ax['P'].set_ylim(.1,.3)
# ax['P'].set_yticks([.1,.15,.2,.25,.3])
#


ax['P'].set_ylim(0.,.55)
ax['A'].yaxis.set_major_formatter(plt.ScalarFormatter())
ax['B'].yaxis.set_major_formatter(plt.ScalarFormatter())
ax['A'].yaxis.set_minor_formatter(plt.ScalarFormatter())
ax['B'].yaxis.set_minor_formatter(plt.ScalarFormatter())
#
fig.subplots_adjust(bottom=.22)
#
fig.savefig('figures/rendered/EPres_camera.pdf')
fig.savefig('figures/rendered/EPres_camera.png',dpi=300)
plt.show()
