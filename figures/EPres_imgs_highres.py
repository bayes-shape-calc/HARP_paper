## 23

import common_figures as cf
import numpy as np
import matplotlib.pyplot as plt


deposit_date = cf.load_depositdate('./all_results.hdf5')
resolution = cf.load_resolution('./all_results.hdf5')
P_res = cf.load_P_res('./all_results.hdf5')
pdbids = cf.load_pdbids('./all_results.hdf5')

num_particles = cf.load_numparticles('./all_results.hdf5')
print('loaded')

for i in range(len(P_res)):
	P_res[i][np.isnan(P_res[i])] = 0.

### apply year cutoff
ycut = 2018
rescutoff = 3.2
keep =[False for _ in range(len(deposit_date))]
for i in range(len(deposit_date)):
	y,m,d = deposit_date[i].split('-')
	if (int(y) >= ycut) and resolution[i] <= rescutoff:
		keep[i] = True

keep = np.bitwise_and(keep,np.bitwise_not(np.isnan(num_particles)))

deposit_date = [deposit_date[i] for i in range(len(deposit_date)) if keep[i]]
P_res = [P_res[i] for i in range(len(P_res)) if keep[i]]
pdbids = [pdbids[i] for i in range(len(pdbids)) if keep[i]]
pdbids = np.array(pdbids)
num_particles = num_particles[keep]



xorder = num_particles.argsort()
ngroup = 100
n = xorder.size//ngroup + 1
P_ress = []
rr = []
for i in range(n):
	if i == 0:
		P = []
		rmin = 0
	else:
		if rmax - rmin  < 1e-10:
			pass
		else:
			P_ress.append(P)
			rr.append((rmax+rmin)*.5)
			rmin = num_particles[xorder[i*ngroup]]
			P = []
	for j in range(ngroup):
		ind = i*ngroup+j
		if ind < xorder.size:
			P.append(P_res[xorder[ind]])
			rmax = num_particles[xorder[ind]]
rr = np.array(rr)


# # ps,fig,ax = cf.process_sets_indiv(x,P_res_months)
ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_ress,'figures/models/hmodel_nparticles_highres.hdf5',nres=5)



fig,ax = cf.make_fig_set_ab(rr,mu_as,mu_bs,tau_as,tau_bs,covs)

rmax = np.log10(rr[-1])//1+1
rmin = np.log10(rr[0])//1
rx = np.logspace(rmin,rmax,int(rmax-rmin+1)).astype('int')
for aa in fig.axes:
	aa.set_xticks(rx)
	aa.set_xticklabels(rx,rotation=0)
	aa.set_xlim(rr[0]*.9,rr[-1]*1.1)
	# aa.set_xlim(5,65)
	# aa.xaxis.major.formatter._useMathText = True
	# aa.ticklabel_format(axis='x',style='sci')
	aa.set_xscale('log')
	
	
ax['P'].set_ylim(0,1.)
ax['P'].set_xlabel(r'Number of Particles')
ax['B'].set_xlabel(r'Number of Particles')


fig.tight_layout()
fig.subplots_adjust(bottom=.22)
#
fig.savefig('figures/rendered/EPres_nparticles_highres.pdf')
fig.savefig('figures/rendered/EPres_nparticles_highres.png',dpi=300)
plt.close()
