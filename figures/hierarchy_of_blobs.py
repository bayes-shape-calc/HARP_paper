## 23

import harp
import numpy as np

#### Preparation
pdbid = '7a4m'
basedir = '../testdb'

levels = [0,1,2,3]
level = levels[0]



#### I/O
success, path_mol, path_density, flag_density = harp.io.rcsb.get_pdb(pdbid,basedir,False,False,print)
mol = harp.molecule.load(path_mol,True)
grid,density = harp.density.load(path_density)
print('--> Loaded %s'%(pdbid))
print('--> Grid: %s, %s, %s'%(str(grid.origin),str(grid.nxyz),str(grid.dxyz)))


## upscale
grid.nxyz *= 2
grid.dxyz /= 2.



#### atomic
if level == 0:	
	model = harp.models.density_atoms(grid,mol.xyz,weights=np.ones(mol.natoms),sigma=0.3)


#### primary
elif level == 1:
	model = np.zeros_like(density)
	adf = 0.3
	coms = []
	weights = []
	blobs = []
	for residue in mol.unique_residues:
		subresidue = mol.get_residue(residue)
		r = np.sqrt(np.sum((subresidue.xyz-subresidue.com()[None,:])**2.,axis=1))
		E_r = np.mean(r)
		E_rr = np.mean(r*r)
		blob = np.sqrt(adf**2.+E_rr-E_r**2.)
		coms.append(subresidue.xyz.mean(0))
		weights.append(subresidue.natoms)
		blobs.append(blob)
	model = harp.models._render_model(grid.origin,grid.dxyz,grid.nxyz,np.array(coms),np.array(weights),np.array(blobs),5,.5)

	
#### secondary
elif level == 2:
	from sklearn.cluster import KMeans
	nclusters = 20
	kmeans = KMeans(n_clusters=nclusters).fit(mol.xyz)
	clusters = kmeans.cluster_centers_

	coms = []
	for residue in mol.unique_residues:
		subresidue = mol.get_residue(residue)
		coms.append(subresidue.com())
	coms = np.array(coms)

	_groups = kmeans.predict(coms)
	print(_groups.shape)
	groups = np.zeros(mol.natoms)
	for residue in mol.unique_residues:
		groups[mol.resid == residue] = _groups[residue-1]
	weights = np.array([np.sum(groups ==i) for i in range(nclusters)])
	coms = np.array([mol.xyz[groups==i].mean(0) for i in range(nclusters)])
	blobs = np.zeros(nclusters)
	adf = .5
	for i in range(nclusters):
		molset = mol.get_set(groups == i)
		r = np.sqrt(np.sum((molset.xyz-molset.com()[None,:])**2.,axis=1))
		E_r = np.mean(r)
		E_rr = np.mean(r*r)
		blobs[i] = np.sqrt(adf**2.+E_rr-E_r**2.)
	model = harp.models._render_model(grid.origin,grid.dxyz,grid.nxyz,coms,weights,blobs,5,.5)


##### tertiary
elif level == 3:
	adf =.3
	r = np.sqrt(np.sum((mol.xyz-mol.com()[None,:])**2.,axis=1))
	E_r = np.mean(r)
	E_rr = np.mean(r*r)
	blob = np.sqrt(adf**2.+E_rr-E_r**2.)
	model = harp.models.density_point(grid,mol.com(),weight=mol.natoms,sigma=blob)



#### RENDER
print(model.sum(),density.sum(),model.min(),model.max(),model.mean())
import blobview
options = blobview.default_options
options['iso_thresh']=model.max()*.01
blobview.view(density=model, grid=grid, mol=mol,options=options)

