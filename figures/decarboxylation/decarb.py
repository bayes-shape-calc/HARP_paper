import harp
import numpy as np
import matplotlib.pyplot as plt

pdbids = ['6cl7','6cl8','6cl9','6cla','6clb']
emdbids = ['7490','7491','7492','7493','7494']
dose = ['0.86 e- / A^2','2.60 e- / A^2','4.30 e- / A^2','6.00 e- / A^2','7.80 e- / A^2']
resolution = ['1.71 A','2.00 A','2.20 A','2.80 A','3.20 A']
# ASP170
# Glu153
# cys139-cys228


from harp import density as dc
from harp import models
from harp import evidence
harp.models.use_python()

def calc_lnev(grid, data, subresidue, adfs, subgrid_size=8., sigmacutoff=5, offset=0.5, ):
	
	atom_weights = np.array([.05,1.,1.,1.,2.,2.])
	atom_types = np.array(['H','C','N','O','P','S'])
	
	nadf = adfs.size
	# nblob = blobs.size
	ncalc = nadf

	## this includes the hydrogens not used in the calculation. make sure to keep the +,- down below!
	record_prob_good = np.zeros((subresidue.natoms))
	record_ln_ev = np.zeros((subresidue.natoms,ncalc))

	## if there are no atoms left in the residue, don't do the calculation
	if subresidue.natoms == 0:
		return record_prob_good,record_ln_ev

	com = subresidue.com()

	## only run the calculations using the local neighborhood (a la BITS definition)
	subgrid = dc.subgrid_center(grid,com,subgrid_size)
	subdata = dc.subgrid_extract(grid,data,subgrid).astype('double')

	## if the position is outside of the density grid, we can't calculate anything
	if subdata.size == 0:
		record_prob_good += 0. # model priors set to zero if outside.
		record_ln_ev -= 2*evidence.lnprior_factor_location + evidence.lnprior_factor_scale ## keep the + and - signs!
		return record_prob_good,record_ln_ev

	#### Initialize atom weights
	if atom_weights is None or atom_types is None:
		## usually normalized to carbon. Could be anything b/c a shape analysis, but needs to be something
		raise Exception('no weights provided!')

	weights = np.ones(subresidue.natoms)
	for ati in range(atom_types.size):
		weights[subresidue.element == atom_types[ati]] = atom_weights[ati] ## assign weights
	if subresidue.occupancy.sum() > 0:
		## if multiple conformations, usually the occupancy is split, so do this. doesn't really matter unless trying to compare absolute values between residues, but that's non-sense anyway because the local neighborhoods might be different sizes.
		weights *= subresidue.occupancy

	#### Calculations

	ln_evidence_out = np.zeros((ncalc))

	## Run the atomic model - adfs
	for i in range(nadf):
		atomic_model = models.density_atoms(subgrid, subresidue.xyz, weights, adfs[i], sigmacutoff, offset)
		ln_evidence_out[i] = evidence.ln_evidence(atomic_model, subdata)
		
		
	lnpriors = np.ones(ncalc)
	
	if nadf > 1:
		dlnx = np.log(adfs[-1])-np.log(adfs[0])
		adfbounds = (adfs[1:]+adfs[:-1])/2.
		lnpriors[0] *= (np.log(adfbounds[0]) - np.log(adfs[0]))/dlnx
		lnpriors[nadf-1] *= (np.log(adfs[-1]) - np.log(adfbounds[-1]))/dlnx
		if nadf > 2:
			lnpriors[1:nadf-1] *= (np.log(adfbounds[1:]) - np.log(adfbounds[:-1]))/dlnx
			
	return ln_evidence_out+lnpriors
	
	
	

fdir = './figures/decarboxylation/'

report = {}
for i in range(len(pdbids)):
	cif_local = harp.io.rcsb.path_cif_local(pdbids[i],fdir)
	cif_ftp = harp.io.rcsb.path_cif_ftp(pdbids[i],fdir)
	map_local = harp.io.rcsb.path_map_local(emdbids[i],fdir)
	map_ftp = harp.io.rcsb.path_map_ftp(emdbids[i],fdir)
	
	harp.io.rcsb.download_wrapper(cif_ftp,cif_local,overwrite=False)
	harp.io.rcsb.download_wrapper(map_ftp,map_local,overwrite=False)
	
	
	mol = harp.molecule.load(cif_local,authid=True)
	grid,density = harp.density.load(map_local)
	adfs,_ = harp.bayes_model_select.gen_adfsblobs(adf_low=.25,adf_high=1.,adf_n=50)

	
	for resid in mol.unique_residues:
		subresidue = mol.get_residue(resid)
		resname = subresidue.resname[0]

		if resname == 'GLU': ## ['CD','OE1','OE2']
			#### Would do this but there are no hydrogens in this molecule... so what's the point?
			keep = np.bitwise_not(np.isin(subresidue.atomname,['OE1','OE2']))
			decarboxylated = subresidue.get_set(keep)
			cap = decarboxylated.atomname == 'CD'
			decarboxylated.resname[cap] = 'HCAP'
			decarboxylated.element[cap] = 'H'
			
		elif resname == 'ASP': ## ['CG','OD1','OD2']
			#### same as above...
			keep = np.bitwise_not(np.isin(subresidue.atomname,['OD1','OD2']))
			decarboxylated = subresidue.get_set(keep)
			cap = decarboxylated.atomname == 'CG'
			decarboxylated.resname[cap] = 'HCAP'
			decarboxylated.element[cap] = 'H'
			
		else:
			continue
			
		
		patomic, _ = harp.bayes_model_select.bms_molecule(grid,density,subresidue,emit=lambda x:None)
		patomicminus, _ = harp.bayes_model_select.bms_molecule(grid,density,decarboxylated,emit=lambda x:None)
		lnev_carb = calc_lnev(grid, density, subresidue, adfs=adfs)
		lnev_decarb = calc_lnev(grid, density, decarboxylated, adfs=adfs)
	
		lnev = np.append(lnev_carb,lnev_decarb)
		
	
		lnev -= lnev.max()
		R = np.exp(lnev) ## equal a priori priors
		prob = np.nansum(R[adfs.size:])/np.nansum(R)

		if not resid in report:
			report[resid] = []
		report[resid].append([pdbids[i],resid,resname,patomic[0],patomicminus[0],prob])


out = ''
for resid in report.keys():
	out += '%s%d\n'%(report[resid][0][2],resid)
	out += 'PDBID, Dose, Resolution, P_atomic(+C02), P_atomic(-C02), Prob(-CO2|+CO2 or -CO2)\n'
	out += '========================================\n'
	for line in report[resid]:
		pind = pdbids.index(line[0])
		out += '%s: %s, %s; %.4f, %.4f, %.4f\n'%(line[0],dose[pind],resolution[pind],line[-3],line[-2],line[-1])
	out += '\n'

with open(fdir+'Decarboxylation_ProteinaseK.txt','w') as f:
	f.write(out)



doses = [.86,2.6,4.3,6.,7.8] 

datas = []
for resid in report.keys():
	data = []
	for i in range(len(report[resid])):
		line = report[resid][i]
		data.append([doses[i],line[-1]])
	data = np.array(data)
	datas.append(data)
datas = np.array(datas)

mu = np.mean(datas[:,:,1],axis=0)
sem = np.std(datas[:,:,1],axis=0)/np.sqrt(0.+datas.shape[0])

plt.errorbar(doses,mu,yerr=sem,linestyle='None',marker='o',color='tab:blue')
plt.xlabel(r'Dose ($e^{-} \AA^{-2}$)')
plt.ylabel(r'Average $P_{-CO_2}$ for Glu or Asp')
plt.xlim(0,8.)
plt.ylim(0,1)
plt.title('Proteinase K')
plt.savefig('figures/rendered/fig_decarb_proteinasek.pdf')
plt.savefig('figures/rendered/fig_decarb_proteinasek.png')
plt.close()

