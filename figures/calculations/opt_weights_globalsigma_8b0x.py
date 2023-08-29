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
	atom_types = np.array(['C','N','O','Q','S','P','MG','K'])
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
pdbid = '8b0x'
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

# mol = mol.get_set(np.isin(mol.element,np.array(['C','N','O','Q','P','S','MG','K','ZN'])))
mol = mol.get_set(np.bitwise_not(np.bitwise_and(np.bitwise_not(np.isin(mol.element,['MG','K'])),mol.hetatom)))
# mol = mol.remove_hetatoms()


phosphate_atoms = mol.bool_NA()[0]
phosphates_oxygen = np.bitwise_and(phosphate_atoms,mol.element=='O')
mol.element[phosphates_oxygen] = 'Q'

# mol = mol.get_chain('E') ## probably want to include proteins if also doing S
# 16S: E, 23S: BA

## debug reporting
print(np.sum(mol.element=='Q'))
print(np.unique(mol.atomname))
print(np.unique(mol.element))


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




adfmap = harp.bayes_model_select.optimal_global_adf(grid,density,mol,5,.5,sigmas=np.linspace(.45,.55,21))
# adfmap = 0.48
print(adfmap)
atom_weights = np.array([1., 1.14197459, 0.86538742, 0.85803704, 1.12661538, 2.0084891, 1.93298994, 2.52722707])
out = minimize(minfxn,x0=atom_weights[1:],args=(grid,mol,density,adfmap),method='Nelder-Mead')
print(out)
print(out.x)
# for _ in range(10):
# 	out = minimize(minfxn,x0=out.x,args=(grid,mol,density,adfmap),method='Nelder-Mead')
# 	print(out)
# 	print(out.x)


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
