## 23

import harp
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
harp.models.use_c()


basedir = '../testdb'
for pdbid in ['6j6j','6j6k','7a4m','6z6u','8b0x']:
	# if pdbid != '8b0x':
	# 	continue

	## Preparation
	success, path_mol, path_density, flag_density = harp.io.rcsb.get_pdb(pdbid,basedir,False,False,print)
	mol = harp.molecule.load(path_mol,True)
	grid,density = harp.density.load(path_density)
	print('--> Loaded %s'%(pdbid))
	print('--> Grid: %s, %s, %s'%(str(grid.origin),str(grid.nxyz),str(grid.dxyz)))


	## select atoms
	mol = mol.remove_hetatoms()
	# mol = mol.dehydrogen()


	## shrink data
	mins = mol.xyz.min(0)-8.
	maxes = mol.xyz.max(0)+8.
	nmins = ((mins-grid.origin)//grid.dxyz).astype('int')
	nmaxes = ((maxes-grid.origin)//grid.dxyz + 1).astype('int')
	nmins[nmins<0] = 0
	nmaxes[nmaxes>grid.nxyz] = grid.nxyz[nmaxes>grid.nxyz]
	grid.nxyz = nmaxes-nmins
	grid.origin = nmins*grid.dxyz+grid.origin
	density = density[nmins[0]:nmaxes[0],nmins[1]:nmaxes[1],nmins[2]:nmaxes[2]].copy()
	print('Shrink to:',grid.nxyz)



	## prep weights
	weights_theory = np.ones(mol.natoms)
	weights_uniform = np.ones(mol.natoms)
	weights_opt = np.ones(mol.natoms)

	atom_weights = np.array([0.0457071, 1., 1.18519213, 1.32159026, 3.72879692, 4.21957656])
	atom_types = np.array(['H','C','N','O','P','S'])
	for ati in range(atom_types.size):
		weights_theory[mol.element == atom_types[ati]] = atom_weights[ati]

	# atom_types_opt = np.array(['H','C','N','O','S','P','MG','K'])
	# atom_weights_opt = np.array([ 0., 1., 1.14197459, 0.86538742, 1.12661538, 2.0084891, 1.93298994, 2.52722707]) ## 8b0x

	# atom_types_opt = np.array(['H','C','N','O','S'])
	# atom_weights_opt = np.array([0.07429156, 1., 1.06487857, 1.00059229, 1.71836897]) ## 7a4m


	## composite conclusions from 7a4m, 8b0x, 6z6u
	atom_types_opt = np.array(['H','C','N','O','P','S'])
	atom_weights_opt = np.array([ 0.05, 1., 1., 1., 2., 2.]) ## composite

	for ati in range(atom_types_opt.size):
		weights_opt[mol.element == atom_types_opt[ati]] = atom_weights_opt[ati]

	weights_uniform[mol.element=='H'] = 0.

	weights_theory *= mol.occupancy
	weights_uniform *= mol.occupancy
	weights_opt *= mol.occupancy

	## compare offset curves
	n = 200
	adfs = np.linspace(0.005,1.,n)


	lnev_theory = np.zeros_like(adfs)
	lnev_uniform = np.zeros_like(adfs)
	lnev_opt = np.zeros_like(adfs)
	import tqdm
	for i in tqdm.tqdm(range(adfs.size)):
		model = harp.models.density_atoms(grid, mol.xyz, weights_theory, adfs[i], 5, .5)
		lnev_theory[i] = harp.evidence.ln_evidence(model, density)
		model = harp.models.density_atoms(grid, mol.xyz, weights_uniform, adfs[i], 5, .5)
		lnev_uniform[i] = harp.evidence.ln_evidence(model, density)
		model = harp.models.density_atoms(grid, mol.xyz, weights_opt, adfs[i], 5, .5)
		lnev_opt[i] = harp.evidence.ln_evidence(model, density)

	plt.figure()
	plt.axvline(x=.056,color='k',linestyle='--',lw=.8)
	plt.axvline(x=.25,color='k',linestyle='--',lw=.8)
	plt.axvline(x=adfs[np.argmax(lnev_theory)],color='k',linestyle='--',lw=.8)

	# fullmodel = harp.models.density_atoms(grid,mol.xyz.mean(0)[None],np.array([weights_opt.sum(),]),mol.xyz.std(0).mean(),5,.5)
	# lnevfull = harp.evidence.ln_evidence_scipy(fullmodel,density)

	plt.ion()
	plt.plot(adfs,lnev_theory,label='Theoretical Weights')
	plt.plot(adfs,lnev_uniform,label='Uniform Heavy Weights')
	plt.plot(adfs,lnev_opt,label='Optimized Weights')
	plt.legend()
	plt.title('%s'%(pdbid.upper()))
	plt.xlim(0,1)
	plt.xlabel(r'$\sigma_{ADF}$')
	plt.ylabel('ln(shape evidence)')
	plt.tight_layout()
	plt.savefig('figures/rendered/global_adfweights_%s.pdf'%(pdbid))
	plt.savefig('figures/rendered/global_adfweights_%s.png'%(pdbid),dpi=300)
	plt.show()
