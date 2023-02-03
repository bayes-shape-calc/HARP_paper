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
xns = np.arange(len(res_ns))


cutoff_year = 1900

for cutoff_resolution in [2.,2.25,2.5,2.75,3.,3.25,3.5,3.75,4.]:

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
				pp[~np.isfinite(pp)] = 0.
				pp[np.isnan(pp)] = 0.

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
				# for j in range(len(res_ns)):
				# 	keep = residues[i] == res_ns[j]
				# 	if np.sum(keep)>0:
				# 		P_ns[j].append(pp[keep])


	####
	print('****** %f ******'%(cutoff_resolution))

	keep = np.array([len(pi) > 1 for pi in P_aa])
	P_aa = [P_aa[i] for i in range(len(keep)) if keep[i]]
	ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_aa,nres=20)
	fig,ax = cf.make_fig_set_ab(x_aa[keep],mu_as,mu_bs,tau_as,tau_bs,covs)
	for aa in fig.axes:
		aa.set_xlim(x_aa.min(),x_aa.max())
		aa.set_xticks(x_aa)
		aa.set_xticklabels(res_aa,rotation=90)
	ax['P'].grid(axis='x')
	ax['A'].yaxis.set_major_formatter(plt.ScalarFormatter())
	ax['B'].yaxis.set_major_formatter(plt.ScalarFormatter())
	ax['A'].yaxis.set_minor_formatter(plt.ScalarFormatter())
	ax['B'].yaxis.set_minor_formatter(plt.ScalarFormatter())
	fig.savefig('figures/rendered/EPres_residue_aa_%.2f.png'%(cutoff_resolution),dpi=300)
	fig.savefig('figures/rendered/EPres_residue_aa_%.2f.pdf'%(cutoff_resolution))
	plt.close()

	# keep = np.array([len(pi) > 1 for pi in P_dna])
	# P_dna = [P_dna[i] for i in range(len(keep)) if keep[i]]
	# ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_dna,nres=20)
	# fig,ax = cf.make_fig_set_ab(x_dna[keep],mu_as,mu_bs,tau_as,tau_bs,covs)
	# for aa in fig.axes:
	# 	aa.set_xlim(x_dna.min(),x_dna.max())
	# 	aa.set_xticks(x_dna)
	# 	aa.set_xticklabels(res_dna,rotation=90)
	# ax['P'].grid(axis='x')
	# fig.savefig('figures/rendered/fig_residue_dna_%.2f.png'%(cutoff_resolution),dpi=300)
	# fig.savefig('figures/rendered/fig_residue_dna_%.2f.pdf'%(cutoff_resolution))
	# plt.close()
	#
	# keep = np.array([len(pi) > 1 for pi in P_rna])
	# P_rna = [P_rna[i] for i in range(len(keep)) if keep[i]]
	# ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_rna,nres=20)
	# fig,ax = cf.make_fig_set_ab(x_rna[keep],mu_as,mu_bs,tau_as,tau_bs,covs)
	# for aa in fig.axes:
	# 	aa.set_xlim(x_rna.min(),x_rna.max())
	# 	aa.set_xticks(x_rna)
	# 	aa.set_xticklabels(res_rna,rotation=90)
	# ax['P'].grid(axis='x')
	# fig.savefig('figures/rendered/fig_residue_rna_%.2f.png'%(cutoff_resolution),dpi=300)
	# fig.savefig('figures/rendered/fig_residue_rna_%.2f.pdf'%(cutoff_resolution))
	# plt.close()
