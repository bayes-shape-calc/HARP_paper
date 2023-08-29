## 23

import harp
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import os
harp.models.use_c()

# from matplotlib.ticker import ScalarFormatter
# plt.rcParams['axes.autolimit_mode'] = 'round_numbers'

import multiprocessing
pdbids = ['7A4M','7T2G','6J6J','6D6V']
resolutions = [1.22,2.5,3.2,4.8]
nsi = 200
dfmaxi = 10.
theta = [[pdbids[i],resolutions[i],nsi,dfmaxi] for i in range(len(pdbids))]
# theta = [theta[2],]

def proc(thetai):
	print(thetai)
	pdbid,resolution,n,dfmax = thetai
	basedir = '../testdb'

	## Preparation
	success, path_mol, path_density, flag_density = harp.io.rcsb.get_pdb(pdbid,basedir,False,False,print)
	mol = harp.molecule.load(path_mol,True)
	grid,density = harp.density.load(path_density)
	print('--> Loaded %s'%(pdbid))
	print('--> Grid: %s, %s, %s'%(str(grid.origin),str(grid.nxyz),str(grid.dxyz)))


	# ## shrink data
	# mins = mol.xyz.min(0)-8.
	# maxes = mol.xyz.max(0)+8.
	# nmins = ((mins-grid.origin)//grid.dxyz).astype('int')
	# nmaxes = ((maxes-grid.origin)//grid.dxyz + 1).astype('int')
	# nmins[nmins<0] = 0
	# nmaxes[nmaxes>grid.nxyz] = grid.nxyz[nmaxes>grid.nxyz]
	# grid.nxyz = nmaxes-nmins
	# grid.origin = nmins*grid.dxyz+grid.origin
	# density = density[nmins[0]:nmaxes[0],nmins[1]:nmaxes[1],nmins[2]:nmaxes[2]].copy()
	# print('Shrink to:',grid.nxyz)

	# mol = mol.remove_hetatoms()
	# mol = mol.dehydrogen()
	atom_types = np.array(['H','C','N','O','P','S'])
	atom_weights = np.array([ 0.05, 1., 1., 1., 2., 2.]) ## composite
	weights = np.zeros(mol.natoms)
	for ati in range(atom_types.size):
		weights[mol.element == atom_types[ati]] = atom_weights[ati]


	## compare offset curves
	adfs = np.logspace(np.log10(0.01),np.log10(4),n)
	adfs = np.concatenate([adfs[:-1],np.logspace(np.log10(adfs[-1]),np.log10(dfmax),5)])


	fname = 'figures/models/adfs_%s.npy'%(pdbid)
	flag = False
	if os.path.exists(fname):
		d = np.load(fname)
		if np.all(adfs == d[0]):
			adfs,lnev_face = d
			flag = True
	print(fname,'cached',flag)
	if not flag:
		lnev_face = np.zeros_like(adfs)
		# lnev_edge = np.zeros_like(adfs)
		# import tqdm
		# for i in tqdm.tqdm(range(adfs.size)):
		for i in range(adfs.size):
			model = harp.models.density_atoms(grid, mol.xyz, weights,adfs[i], 5, .5)
			lnev_face[i] = harp.evidence.ln_evidence(model, density)
			# model = harp.models.density_atoms(grid, mol.xyz, weights,adfs[i], 5, .0)
			# lnev_edge[i] = harp.evidence.ln_evidence(model, density)
		np.save(fname,np.array([adfs,lnev_face]))

	plt.figure(figsize=(2,1.5),dpi=600)
	# plt.axvline(x=.056,color='k',linestyle='--',lw=.8)
	# plt.axvline(x=.25,color='k',linestyle='--',lw=.8)
	plt.axvline(x=adfs[np.argmax(lnev_face)],color='Gray',linestyle='-',lw=1.)

	plt.ion()
	plt.plot(adfs,lnev_face,color='tab:red',lw=1.25)#,label='Face-centered')

	plt.xlabel(r'$\sigma_{0}$ ($ \AA $)',fontsize=6)
	plt.ylabel('Log Probability',fontsize=6)
	plt.xscale('log')
	plt.xlim(adfs[0],adfs[-1])
	plt.xlim(adfs[0],10.)

	props = dict( facecolor='None',edgecolor='None',alpha=0)
	ax = plt.gca()
	ax.text(0.05, 0.925, '%s\n%s'%(pdbid,r'%s $\AA$'%(resolution)), transform=ax.transAxes, fontsize=6, color='Gray',verticalalignment='top', bbox=props)
	

	ax.yaxis.set_major_locator(plt.MaxNLocator(6))
	ax.tick_params(axis='both', which='major', labelsize=6)
	from matplotlib import ticker
	fmt = ticker.ScalarFormatter(useOffset=True,useMathText=True)
	# fmt.set_scientific(False)
	# fmt.set_powerlimits((-1,1))
	ax.yaxis.set_major_formatter(fmt)
	ax.yaxis.get_offset_text().set_fontsize(6)
	# ax.yaxis.set_label_coords(-.225, .5)
	ax.xaxis.set_label_coords(.5, -.15)
	
	plt.subplots_adjust(left=.25,right=.965,bottom=.2,top=.9)
	# plt.tight_layout()
	plt.savefig('figures/rendered/global_adfscan_%s.pdf'%(pdbid))
	plt.savefig('figures/rendered/global_adfscan_%s.png'%(pdbid),dpi=600)
	plt.close()
	print('done',pdbid)

def run_stuff():
	with multiprocessing.Pool(processes=4) as pool:
		out = pool.map(proc,theta)

if __name__=='__main__':
	run_stuff()
