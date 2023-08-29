## 23

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

		mass = np.zeros(subresidue.natoms)
		mass[subresidue.element == 'C'] = 12.011
		mass[subresidue.element == 'N'] = 14.007
		mass[subresidue.element == 'O'] = 15.999
		mass[subresidue.element == 'P'] = 30.974
		mass[subresidue.element == 'S'] = 32.065
		mass[subresidue.element == 'H'] = 1.0078
		
		r2 = np.sum((subresidue.xyz-subresidue.com()[None,:])**2.,axis=1)
		r = np.sqrt(r2)
		
		##https://en.wikipedia.org/wiki/Radius_of_gyration
		rg = np.sqrt(np.sum(mass*r2)/np.sum(mass))
		
		
		# E_r = np.mean(r)
		# E_rr = np.mean(r*r)
		# adf = 0.#0.32
		# blob = np.sqrt(adf**2.+E_rr-E_r**2.)
		#
		# if blob > 10 or blob < .2:
		# 	print(subresidue.xyz)
		# 	print(E_r,E_rr,blob)
		# 	print(subresidue.com())
		# 	raise Exception('stop')

		resname = subresidue.resname[0]
		if not resname in out:
			out[resname] = [rg,]
		else:
			out[resname].append(rg)


mu = []
mu_min = []
mu_max = []
keys = np.array(list(out.keys()))
keys.sort()

for key in keys:
	mu.append(np.mean(out[key]))
	mu_min.append(np.min(out[key]))
	mu_max.append(np.max(out[key]))
	print('%.4f, #%s'%(mu[-1],key))

print(np.mean(mu))
print(np.min(mu_min))
print(np.max(mu_max))
