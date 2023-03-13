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






initiates_aa = np.array([
[-1.816954, -1.676226, 0.999041, 3.720686], ### 1189.394 
[-1.465683, -2.420246, 0.280349, 14.071142], ### 1403.510 
[-1.842429, -2.134395, 0.476395, 6.364033], ### 974.578 
[-2.008048, -1.806475, 1.832338, 3.897114], ### 883.342 
[-2.06189,  -1.74083,  9.18947,  2.92470], ### 280.375
[-1.864313, -2.300063, 0.835012, 10.702769], ### 893.905 
[-1.956384, -2.045688, 0.684753, 5.590446], ### 1139.322 
[-1.965107, -1.753363, 1.247194, 4.477114], ### 1153.528 
[-1.58648,  -2.39776,  0.28433, 20.50844], ### 542.328
[-1.321205, -2.104695, 0.338108, 6.326351], ### 1034.617 
[-1.304863, -2.208597, 0.303266, 6.613471], ### 1814.754 
[-1.696860, -2.348307, 0.402396, 11.812987], ### 1366.889 
[-1.449717, -2.186855, 0.256598, 7.843238], ### 462.048 
[-0.976801, -2.374033, 0.155823, 10.581196], ### 1062.016 
[-1.716281, -2.036772, 0.480945, 6.637983], ### 807.701 
[-1.932125, -1.712408, 1.791262, 3.567667], ### 1018.351 
[-1.720739, -1.856834, 0.554167, 4.038648], ### 961.491 
[-0.937446, -2.422343, 0.145753, 21.237133],
[-1.076941, -2.426638, 0.159363, 14.186267], ### 950.797 
[-1.481644, -1.929240, 0.429856, 4.451450], ### 1122.131 
])


# initiates_aa = None


cutoff_year = 1900

# for cutoff_resolution in [2.,2.25,2.5,2.75,3.,3.25,3.5,3.75,4.]:
for cutoff_resolution in [3.,]:

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
				# for j in range(len(res_ns)):
				# 	keep = residues[i] == res_ns[j]
				# 	if np.sum(keep)>0:
				# 		P_ns[j].append(pp[keep])


	####
	print('****** %f ******'%(cutoff_resolution))
	
	
	# ### TEST LIMITS
	# for i in range(len(P_aa)):
	# 	if len(P_aa[i]) > 500:
	# 		P_aa[i] = P_aa[i][:500]
	

	keep = np.array([len(pi) > 1 for pi in P_aa])
	P_aa = [P_aa[i] for i in range(len(keep)) if keep[i]]
	# ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_aa,'figures/models/hmodel_residue_%.2f.hdf5'%(cutoff_resolution),nres=20,initiates=initiates_aa)

	ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_aa, 'figures/models/hmodel_residue_%.2f.hdf5'%(cutoff_resolution),nres=20, initiates=initiates_aa)
	fig,ax = cf.make_fig_set_abtautoo(x_aa[keep],mu_as,mu_bs,tau_as,tau_bs,covs)
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
