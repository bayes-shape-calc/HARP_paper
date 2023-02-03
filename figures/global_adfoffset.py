## 23

import harp
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
harp.models.use_c()


basedir = '../testdb'
for pdbid in ['6j6j','6j6k','7a4m','6z6u','8b0x']:
	## Preparation
	success, path_mol, path_density, flag_density = harp.io.rcsb.get_pdb(pdbid,basedir,False,False,print)
	mol = harp.molecule.load(path_mol,True)
	grid,density = harp.density.load(path_density)
	print('--> Loaded %s'%(pdbid))
	print('--> Grid: %s, %s, %s'%(str(grid.origin),str(grid.nxyz),str(grid.dxyz)))


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

	mol = mol.remove_hetatoms()
	# mol = mol.dehydrogen()
	atom_types = np.array(['H','C','N','O','P','S'])
	atom_weights = np.array([ 0.05, 1., 1., 1., 2., 2.]) ## composite
	weights = np.zeros(mol.natoms)
	for ati in range(atom_types.size):
		weights[mol.element == atom_types[ati]] = atom_weights[ati]



	## compare offset curves
	n = 200
	adfs = np.linspace(0.005,1.5,n)


	lnev_face = np.zeros_like(adfs)
	lnev_edge = np.zeros_like(adfs)
	import tqdm
	for i in tqdm.tqdm(range(adfs.size)):
		model = harp.models.density_atoms(grid, mol.xyz, weights,adfs[i], 5, .5)
		lnev_face[i] = harp.evidence.ln_evidence(model, density)
		model = harp.models.density_atoms(grid, mol.xyz, weights,adfs[i], 5, .0)
		lnev_edge[i] = harp.evidence.ln_evidence(model, density)

	plt.figure()
	plt.axvline(x=.056,color='k',linestyle='--',lw=.8)
	plt.axvline(x=.25,color='k',linestyle='--',lw=.8)
	plt.axvline(x=adfs[np.argmax(lnev_face)],color='k',linestyle='--',lw=.8)

	plt.ion()
	plt.plot(adfs,lnev_face,label='Face-centered')
	plt.plot(adfs,lnev_edge,label='Edge-centered')
	plt.legend()
	plt.title('%s'%(pdbid.upper()))
	plt.xlim(0,1)
	plt.xlabel(r'$\sigma_{ADF}$')
	plt.ylabel('ln(shape evidence)')
	plt.tight_layout()
	plt.savefig('figures/rendered/global_adfoffset_%s.pdf'%(pdbid))
	plt.savefig('figures/rendered/global_adfoffset_%s.png'%(pdbid),dpi=300)
	plt.show()
