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


	## compare offset curves
	blobs = np.linspace(.005,.25,40)
	blobs = np.concatenate((blobs[:-1],np.linspace(blobs.max(),2.5,40)))
	if not pdbid in ['6z6u','8b0x']: ## takes too long
		blobs = np.concatenate((blobs[:-1],np.linspace(blobs.max(),10.0,20)))
		blobs = np.concatenate((blobs[:-1],np.linspace(blobs.max(),20.0,20)))
		blobs = np.concatenate((blobs[:-1],np.linspace(blobs.max(),40.0,10)))
	else:
		blobs = np.concatenate((blobs[:-1],np.linspace(blobs.max(),10.0,5)))

	nsuperatoms = 0
	for chain in mol.unique_chains:
		subchain = mol.get_chain(chain)
		nsuperatoms += mol.unique_residues.size

	sa_weights = np.zeros(nsuperatoms)
	sa_coms = np.zeros((nsuperatoms,3))
	i = 0
	for chain in mol.unique_chains:
		subchain = mol.get_chain(chain)
		for residue in subchain.unique_residues:
			subresidue = mol.get_residue(residue)
			weights = np.zeros(subresidue.natoms)
			for ati in range(atom_types.size):
				weights[subresidue.element == atom_types[ati]] = atom_weights[ati]
			sa_weights[i] = weights.sum()
			sa_coms[i] = subresidue.com()
			i+=1

	lnev = np.zeros_like(blobs)
	import tqdm
	for i in tqdm.tqdm(range(blobs.size)):
	# for i in range(blobs.size):
		model = harp.models.density_atoms(grid, sa_coms, sa_weights, blobs[i] , 5, .5)
		lnev[i] = harp.evidence.ln_evidence_scipy(model, density)
		# print(i,lnev[i])


	# fullmodel = harp.models.density_atoms(grid,mol.xyz.mean(0)[None,:],np.array([sa_weights.sum(),]),mol.xyz.std(0).mean(),5,.5)
	# lnevfull = harp.evidence.ln_evidence_scipy(fullmodel,density)
	# print(mol.xyz.std(0).mean())


	plt.figure()
	plt.axvline(x=blobs[np.argmax(lnev)],color='k',linestyle='--',lw=.8)

	plt.ion()
	# plt.plot(blobs,lnev-lnev.max(),label='Global super-atom')
	plt.plot(blobs,lnev,label='Global super-atom')
	plt.legend()
	plt.title('%s'%(pdbid.upper()))
	plt.xlim(0,blobs.max())
	plt.xlabel(r'$\sigma_{blob}$')
	plt.ylabel('ln(shape evidence)')
	plt.tight_layout()
	plt.savefig('figures/rendered/global_blobsigma_%s.pdf'%(pdbid))
	plt.savefig('figures/rendered/global_blobsigma_%s.png'%(pdbid),dpi=300)
	plt.show()
