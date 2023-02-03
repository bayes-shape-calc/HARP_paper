import harp
import numpy as np
import numba as nb
import matplotlib.pyplot as plt


#### Preparation
pdbid = '8b0x'
basedir = '../testdb'

success, path_mol, path_density, flag_density = harp.io.rcsb.get_pdb(pdbid,basedir,False,False,print)
mol = harp.molecule.load(path_mol)
print('--> Loaded %s'%(pdbid))

mol = mol.remove_hetatoms()


out = {}
for chain in mol.unique_chains:
	subchain = mol.get_chain(chain)
	for residue in subchain.unique_residues:
		subresidue = subchain.get_residue(residue)


		r = np.sqrt(np.sum((subresidue.xyz-subresidue.com()[None,:])**2.,axis=1))
		E_r = np.mean(r)
		E_rr = np.mean(r*r)
		adf = 0.32
		blob = np.sqrt(adf**2.+E_rr-E_r**2.)

		if blob > 10 or blob < .2:
			print(subresidue.xyz)
			print(E_r,E_rr,blob)
			print(subresidue.com())
			raise Exception('stop')

		resname = subresidue.resname[0]
		if not resname in out:
			out[resname] = [blob,]
		else:
			out[resname].append(blob)


mu = []
mu_min = []
mu_max = []
for key in out:
	mu.append(np.mean(out[key]))
	mu_min.append(np.min(out[key]))
	mu_max.append(np.max(out[key]))
	print(key,mu[-1])

print(np.mean(mu))
print(np.min(mu_min))
print(np.max(mu_max))
