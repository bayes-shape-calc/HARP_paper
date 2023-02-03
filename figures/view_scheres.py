## 23

import harp
from blobview import viewer as blobview
import numpy as np

### use the q key to toggle the molecule
### use the 8 key to save a .png image
## you have to click the viewer once in order to get it to update


ftp_cif = harp.io.rcsb.path_cif_ftp('7a4m','./')
local_cif = harp.io.rcsb.path_cif_local('7a4m','./')
ftp_map = harp.io.rcsb.path_map_ftp('11638','./')
local_map = harp.io.rcsb.path_map_local('11638','./')

harp.io.rcsb.download_wrapper(ftp_cif,local_cif,overwrite=False)
harp.io.rcsb.download_wrapper(ftp_map,local_map,overwrite=False)

mol = harp.molecule.load(local_cif,True)
grid,density = harp.density.load(local_map)

mol = mol.get_set(mol.resname=='PHE')
i = 4
mol = mol.get_residue(mol.unique_residues[i])
confs = np.unique(mol.conf)
mol1 = mol.get_set(mol.conf==confs[0])
mol2 = mol.get_set(mol.conf==confs[1])
print(mol1.unique_residues)

mask = harp.models.density_atoms(grid,mol.xyz,None,.6,nsigma=5,offset=0.5)
from scipy import ndimage as nd
mask = nd.uniform_filter(mask,5)
mask /= mask.max()
# mask = 0. + (mask > .3)*1.
mask = nd.gaussian_filter(mask,.01)

display = nd.gaussian_filter(density*mask,.01)
grid,display = harp.density.trim_density_to_mol(grid,display,mol)



options = {
	'bgcolor' : 'white',
	'atom_resolution' : 10,
	'atom_radius' : .25,
	'alpha_iso' : .3,
	'alpha_molecule' : .9,
	'alpha_wire' : .9,
	'fov' : 85.,
	'atom_edge_color' : None,
	'bond_color':(.3,.3,.3,.9),
	'bond_radius':.05,
	'iso_thresh' : .01,
	'iso_color' : (0.3,.3,.3),
	'wire_color' : (0.086,.45,.85),
	'quickbond_cutoff': 1.8,
	'width' : 800,
	'height' : 600,
}


print(mol.unique_residues)
v = blobview.viewer(display,grid,mol,probs=None,options=options,verbose=True)

# from PyQt5.QtCore import QTimer
# timer = QTimer()
v.view.camera = v.cam2
# v.view.camera.center=residue.xyz.mean(0)
v.view.camera.center=mol.xyz.mean(0)
v.view.camera.orbit(-122, 44.5)
v.view.camera.distance=6.75#15.
v.view.camera.elevation=40#45.0
v.view.camera.scale_factor = 4.0843380706010315
v.mol_vis.visible = True
v.iso_vis.visible = False

#
colors1 = np.ones((mol1.xyz.shape[0],4))*options['alpha_molecule']
cs = blobview.colors_by_element(mol1)
for i in range(mol1.xyz.shape[0]):
	colors1[i] = blobview.to_rgba(cs[i])
bonds1 = blobview.quick_bonds(mol1,options['quickbond_cutoff'])
v.mol1_vis = blobview.vismol(mol1.xyz,
	bonds = bonds1,
	parent=v.view.scene,
	bond_radius=options['bond_radius'],
	bond_color=options['bond_color'],
	atom_radius=options['atom_radius'],
	edge_color=options['atom_edge_color'],
	cols=options['atom_resolution'],
	rows=options['atom_resolution'],
	face_colors=colors1,
	shading=None
	)

colors2 = np.ones((mol2.xyz.shape[0],4))*options['alpha_molecule']
cs = blobview.colors_by_element(mol2)
for i in range(mol2.xyz.shape[0]):
	colors2[i] = blobview.to_rgba(cs[i])
bonds2 = blobview.quick_bonds(mol2,options['quickbond_cutoff'])
v.mol2_vis = blobview.vismol(mol2.xyz,
	bonds = bonds2,
	parent=v.view.scene,
	bond_radius=options['bond_radius'],
	bond_color=options['bond_color'],
	atom_radius=options['atom_radius'],
	edge_color=options['atom_edge_color'],
	cols=options['atom_resolution'],
	rows=options['atom_resolution'],
	face_colors=colors2,
	shading=None
	)

#
def switch(v):
	if v.mol1_vis.visible:
		v.mol1_vis.visible = False
		v.mol2_vis.visible = True
		v.mol_vis.visible = False
	elif v.mol2_vis.visible:
		v.mol1_vis.visible = False
		v.mol2_vis.visible = False
		v.mol_vis.visible = True
	elif v.mol_vis.visible:
		v.mol_vis.visible=False
		v.mol1_vis.visible=False
		v.mol2_vis.visible=False
	else:
		v.mol1_vis.visible=True
		v.mol2_vis.visible=False
		v.mol_vis.visible=False
	v.view.update()

def printloc(v,x):
	print(v.view.camera.azimuth,v.view.camera.elevation,v.view.camera.scale_factor,v.view.camera.elevation,v.view.camera.roll)
	v.view.camera.viewbox_mouse_event_orig(x)
#
# v.mol2_vis.visible = False
v.view.camera.viewbox_mouse_event_orig = v.view.camera.viewbox_mouse_event
v.view.camera.viewbox_mouse_event =  lambda x: printloc(v,x)
v.hook = lambda : switch(v)

v.remove_filters()
v.use_tight_filters()
# v.iso_vis.attach(v.iso_filter)
v.wire_vis.attach(v.wire_filter)
v.mol_vis.attach(v.mol_filter)
v.mol1_vis.attach(v.mol_filter)
v.mol2_vis.attach(v.mol_filter)
v.view.update()
#
blobview.app.run()
v.view.update()

### use the q key to toggle the molecule
### use the 8 key to save a .png image
## you have to click the viewer once in order to get it to update
