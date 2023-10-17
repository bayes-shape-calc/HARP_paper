import common_figures as cf
import numpy as np
import matplotlib.pyplot as plt


deposit_date = cf.load_depositdate('./all_results.hdf5')
P_res = cf.load_P_res('./all_results.hdf5')
residues = cf.load_residues('./all_results.hdf5')
resolution = cf.load_resolution('./all_results.hdf5')
print('loaded')
for i in range(len(residues)):
	residues[i] = residues[i].astype('U')

## Define Residue Groups
all_residues = np.unique(np.concatenate(residues))
standard_residues = ['A','ALA','ARG','ASN','ASP','C','CYS','DA','DC','DG','DT','G','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','U','VAL']
res_dna = ['DA','DC','DG','DT']
res_rna = ['A','C','G','U']
res_aa = [r for r in standard_residues if (not r in res_dna) and (not r in res_rna)]
res_ns = [r for r in all_residues if not r in standard_residues] ## ns=non-standard

## x coordinates for plots
x_dna = np.arange(4)
x_rna = 4+np.arange(4)
x_aa = 8+np.arange(len(res_aa))
xs = np.concatenate([x_dna,x_rna,x_aa])
x_ns = np.arange(len(res_ns))


initiates_aa = np.array([
[-1.667208, -1.519080, 1.249326, 3.570052], ### 966.426 
[-1.111805, -2.416486, 0.267453, 14.801143], ### 1320.389 
[-1.818346, -2.074335, 0.961617, 6.228729], ### 867.790 
[-1.871158, -1.753872, 1.872073, 4.240304], ### 752.262 
[-1.812251, -1.568608, 1.028937, 3.275973], ### 229.065 
[-1.641930, -2.272378, 0.504857, 10.560398], ### 817.104 
[-1.776310, -2.061064, 0.828518, 6.305567], ### 1010.913 
[-1.812733, -1.634535, 1.252082, 4.433449], ### 951.018 
[-1.461690, -2.353377, 0.170860, 18.137807], ### 484.815 
[-1.186715, -1.994174, 0.405747, 5.732653], ### 870.632 
[-1.054426, -2.124767, 0.314998, 6.255806], ### 1600.872 
[-1.417542, -2.344607, 0.341483, 12.080984], ### 1290.212 
[-1.223276, -2.139754, 0.280910, 7.800832], ### 399.883 
[-0.705159, -2.316938, 0.136702, 8.828582], ### 976.213 
[-1.741547, -1.909842, 0.806894, 3.050437], ### 695.187 
[-1.832606, -1.587168, 2.216017, 3.596666], ### 874.842 
[-1.635222, -1.726448, 0.743041, 3.954489], ### 813.825 
[-1.010496, -2.458828, 0.193184, 9502.021584], ### 371.564 
[-0.675695, -2.370495, 0.115421, 9.182216], ### 878.291 
[-1.360419, -1.782774, 0.570507, 4.113113], ### 913.048 
])
# initiates_aa = None

initiates_dna = np.array([
[-2.324427, -2.611187, 10.128654, 9550.234284], ### 246.647 
[-2.408822, -2.612963, 6742.804414, 9493.776151], ### 261.564 
[-2.392064, -2.597642, 9.732065, 9980.295407], ### 261.856 
[-2.443419, -2.583839, 5179.206167, 9087.417711], ### 234.450 
	
])

initiates_rna = np.array([
[-1.981343, -2.546032, 5.119177, 25.958471], ### 4166.456 
[-2.033606, -2.518322, 7.034091, 18.071054], ### 3735.469 
[-1.988461, -2.559806, 4.607293, 26.157838], ### 5022.512 
[-2.007131, -2.503352, 6.750622, 17.139043], ### 3271.366 
])


cutoff_year = 2018

# for cutoff_resolution in [2.,2.25,2.5,2.75,3.,3.25,3.5,3.75,4.]:
# for cutoff_resolution in [3.2,]:
cutoff_resolution = 3.2

P_dna = [[] for _ in range(len(res_dna))]
P_rna = [[] for _ in range(len(res_rna))]
P_aa  = [[] for _ in range(len(res_aa))]
P_ns  = [[] for _ in range(len(res_ns))]
from tqdm import tqdm
for i in tqdm(range(len(deposit_date))):
	y,m,d = deposit_date[i].split('-')
	if int(y) >= cutoff_year:
		if resolution[i] <= cutoff_resolution:
			pp = P_res[i].copy()
			pp[~np.isfinite(pp)] = 0.#np.random.rand(np.sum(~np.isfinite(pp)))*1e-6
			pp[np.isnan(pp)] = 0.#np.random.rand(np.sum(np.isnan(pp)))*1e-6

			## extract groups
			for j in range(len(res_dna)):
				keep = residues[i] == res_dna[j]
				if np.sum(keep)>0:
					P_dna[j].append(pp[keep])
			for j in range(len(res_rna)):
				keep = residues[i] == res_rna[j]
				if np.sum(keep)>0:
					P_rna[j].append(pp[keep])
			for j in range(len(res_aa)):
				keep = residues[i] == res_aa[j]
				if np.sum(keep)>0:
					P_aa[j].append(pp[keep])
			for j in range(len(res_ns)):
				keep = residues[i] == res_ns[j]
				if np.sum(keep)>0:
					P_ns[j].append(pp[keep])


####
# print('****** %f ******'%(cutoff_resolution))



keep = np.array([len(pi) > 2 for pi in P_dna])
P_dna = [P_dna[i] for i in range(len(keep)) if keep[i]]
ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_dna,'figures/models/hmodel_residue_dna.hdf5',nres=5,maxiter=1000,initiates=initiates_dna)
fig,ax = cf.make_fig_set_abtautoo(x_dna[keep],mu_as,mu_bs,tau_as,tau_bs,covs)
for aa in fig.axes:
	aa.set_xlim(x_dna.min(),x_dna.max())
	aa.set_xticks(x_dna)
	aa.set_xticklabels(res_dna,rotation=90)
ax['P'].grid(axis='x')
ax['P'].set_ylabel(r'$\langle P \rangle_{res}$')
ax['A'].yaxis.set_major_formatter(plt.ScalarFormatter())
ax['B'].yaxis.set_major_formatter(plt.ScalarFormatter())
ax['A'].yaxis.set_minor_formatter(plt.ScalarFormatter())
ax['B'].yaxis.set_minor_formatter(plt.ScalarFormatter())
fig.savefig('figures/rendered/fig_residue_dna.png',dpi=300)
fig.savefig('figures/rendered/fig_residue_dna.pdf')
plt.close()

keep = np.array([len(pi) > 2 for pi in P_rna])
P_rna = [P_rna[i] for i in range(len(keep)) if keep[i]]
ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_rna,'figures/models/hmodel_residue_rna.hdf5',nres=5,maxiter=1000,initiates=initiates_rna)
fig,ax = cf.make_fig_set_abtautoo(x_rna[keep],mu_as,mu_bs,tau_as,tau_bs,covs)
for aa in fig.axes:
	aa.set_xlim(x_rna.min(),x_rna.max())
	aa.set_xticks(x_rna)
	aa.set_xticklabels(res_rna,rotation=90)
ax['P'].grid(axis='x')
ax['P'].set_ylabel(r'$\langle P \rangle_{res}$')
ax['A'].yaxis.set_major_formatter(plt.ScalarFormatter())
ax['B'].yaxis.set_major_formatter(plt.ScalarFormatter())
ax['A'].yaxis.set_minor_formatter(plt.ScalarFormatter())
ax['B'].yaxis.set_minor_formatter(plt.ScalarFormatter())
fig.savefig('figures/rendered/fig_residue_rna.png',dpi=300)
fig.savefig('figures/rendered/fig_residue_rna.pdf')
plt.close()



keep = np.array([len(pi) > 5 for pi in P_ns])
P_ns = [P_ns[i] for i in range(len(keep)) if keep[i]]
res_ns = [res_ns[i] for i in range(len(keep)) if keep[i]]
x_ns = np.arange(len(P_ns))
ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_ns,'figures/models/hmodel_residue_ns.hdf5',nres=20,maxiter=1000)

order = (np.exp(mu_as)/(np.exp(mu_as)+np.exp(mu_bs))).argsort()
mu_as = mu_as[order]
mu_bs = mu_bs[order]
tau_as = tau_as[order]
tau_bs = tau_bs[order]
covs = covs[order]
res_nso = [res_ns[i] for i in order]

fig,ax = cf.make_fig_set_abtautoo(x_ns,mu_as,mu_bs,tau_as,tau_bs,covs)
fig.set_figwidth(10.0*3)
for aa in fig.axes:
	aa.set_xlim(x_ns.min()-.5,x_ns.max()+.5)
	aa.set_xticks(x_ns)
	aa.set_xticklabels(res_nso,rotation=90)
ax['P'].grid(axis='x')
ax['P'].set_ylabel(r'$\langle P \rangle_{res}$')
# ax['A'].yaxis.set_major_formatter(plt.ScalarFormatter())
# ax['B'].yaxis.set_major_formatter(plt.ScalarFormatter())
# ax['A'].yaxis.set_minor_formatter(plt.ScalarFormatter())
# ax['B'].yaxis.set_minor_formatter(plt.ScalarFormatter())
fig.savefig('figures/rendered/fig_residue_ns.png',dpi=300)
fig.savefig('figures/rendered/fig_residue_ns.pdf')
plt.close()
