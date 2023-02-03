import harp
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.optimize import minimize
harp.models.use_c()

##########
fxncalls = 0

def gen_weights(theta,mol):
	weights = np.ones(mol.natoms)
	atom_types = np.array(['C','H','N','O','S'])
	atom_weights = np.zeros(atom_types.size)
	atom_weights[0] = 1. ## normalize to carbon and exclude H
	atom_weights[1:] = theta.copy()
	for ati in range(atom_types.size):
		weights[mol.element == atom_types[ati]] = atom_weights[ati]
	return weights

def minfxn(theta,grid,mol,density,adf):
	if np.any(theta<.01):
		return np.inf
	if np.any(theta>10.):
		return np.inf

	weights = gen_weights(theta,mol)
	model = harp.models.density_atoms(grid, mol.xyz, weights,adf, 5, .5)
	lnev = harp.evidence.ln_evidence(model, density)

	global fxncalls
	fxncalls+=1
	print(fxncalls,theta)
	return -lnev


############


## Preparation
#### octahedral symmetry imposed so this probably doesn't mean much
# pdbid = '6z9f'
## no H 1.56
## all   : 0.44, [0.09705636 1.19052409 0.97999375 2.19740918]
## half 1: 0.44, [0.09396098 1.19374132 0.97766745 2.21689335]
## half 2: 0.44, [0.09392378 1.19376717 0.97793481 2.21692218]
# pdbid = '7A6B' # array([0.09134676, 1.0922205 , 0.96919099, 2.55292514]) no H 1.33, 0.35
pdbid = '6z6u'
## has H 1.25
## all   : 0.32, [0.09763541 1.15061924 1.06215299 2.41961992]
## half 1:
## half 2: 0.32, [0.08680912 1.15161592 1.06024163 2.4314755 ]
# pdbid = '7A6A' #  array([0.09573609, 1.08024604, 0.96858857, 2.44267212]) no H 1.15, 0.31
basedir = '../testdb'

success, path_mol, path_density, flag_density = harp.io.rcsb.get_pdb(pdbid,basedir,False,False,print)
mol = harp.molecule.load(path_mol)
grid,density = harp.density.load(path_density)
print('--> Loaded %s'%(pdbid))
print('--> Grid: %s, %s, %s'%(str(grid.origin),str(grid.nxyz),str(grid.dxyz)))


## Prepare molecule
print(np.unique(mol.atomname))



print(np.unique(mol.element))
print([np.sum(mol.element==el) for el in np.unique(mol.element)])
mol = mol.remove_hetatoms()

print(mol.unique_chains)
chains = mol.unique_chains
nc = chains.size
# mol = mol.get_set(np.isin(mol.chain,chains[::2]))
print(mol.unique_chains)
print([np.sum(mol.element==el) for el in np.unique(mol.element)])


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





adfmap = harp.bayes_model_select.optimal_global_adf(grid,density,mol,5,.5,sigmas=np.linspace(.25,.5,26))
# adfmap = 0.48
print(adfmap)
# atom_weights = np.array([1.,  .1,      1.14204517, 0.86538403, 1.12659025])
# atom_weights = np.array([1.,  0.07429156, 1.06487857, 1.00059229, 1.71836897]) ##7a4m
atom_weights = np.array([1., 0.08527415, 1.15192579, 1.05872492, 2.43547203])
atom_weights = np.array([1., 0.09763541, 1.15061924, 1.06215299, 2.41961992])



out = minimize(minfxn,x0=atom_weights[1:],args=(grid,mol,density,adfmap),method='Nelder-Mead')
print(out)
print(out.x)


#
#
# plt.figure()
# plt.axvline(x=.056,color='k',linestyle='--',lw=.8)
# plt.axvline(x=.25,color='k',linestyle='--',lw=.8)
# plt.axvline(x=adfs[np.argmax(lnev_face)],color='k',linestyle='--',lw=.8)
#
# plt.ion()
# plt.plot(adfs,lnev_face-lnev_face.max(),label='Face-centered')
# plt.plot(adfs,lnev_edge-lnev_face.max(),label='Edge-centered')
# plt.legend()
# plt.title('%s'%(pdbid.upper()))
# plt.xlim(0,1)
# plt.xlabel(r'$\sigma_{ADF}$')
# plt.ylabel('ln(shape evidence)')
# plt.tight_layout()
# plt.savefig('figures/rendered/fig_globalsigma_offset_%s.pdf'%(pdbid))
# plt.savefig('figures/rendered/fig_globalsigma_offset_%s.png'%(pdbid),dpi=300)
# plt.show()
