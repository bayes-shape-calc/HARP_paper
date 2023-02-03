## 23

import harp
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea
import tqdm
harp.models.use_c()

@nb.njit
def hpdi_post(x,p,frac=.95):
	pp = np.sum(p)
	imin = 0
	imax = x.size
	xmax = imax-imin
	for i in range(x.size):
		for j in range(i,x.size):
			pij = np.sum(p[i:j+1])/pp
			if pij >= frac:
				if j-i < xmax:
					imin = i
					imax = j
					xmax = imax-imin
	return np.array((imin,imax),dtype='int')

@nb.njit
def hpdi_post_all(x,ps,frac=.95):
	n = ps.shape[0]
	xlh = np.zeros((n,2))
	xmap = np.zeros(n)
	xmu = np.zeros(n)
	xstd = np.zeros(n)
	for i in range(n):
		q = hpdi_post(x,ps[i],frac)
		xlh[i,0] = x[q[0]]
		xlh[i,1] = x[q[1]]
		xmap[i] = x[np.argmax(ps[i])]
		xmu[i] = np.sum(x*ps[i])/np.sum(ps[i])
		xstd[i] = np.sum(x**2.*ps[i])/np.sum(ps[i])
		xstd[i] -= xmu[i]**2.
		xstd[i] = np.sqrt(xstd[i])
	order = np.argsort(xmap)[::-1]
	return xlh,xmap,xmu,xstd,order


#### Preparation
basedir = '../testdb'
for pdbid in ['7a4m','8b0x']:
	success, path_mol, path_density, flag_density = harp.io.rcsb.get_pdb(pdbid,basedir,False,False,print)
	mol = harp.molecule.load(path_mol,True)
	grid,density = harp.density.load(path_density)
	print('--> Loaded %s'%(pdbid))
	print('--> Grid: %s, %s, %s'%(str(grid.origin),str(grid.nxyz),str(grid.dxyz)))

	mol = mol.remove_hetatoms()

	# #### shrink data for speed
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


	###### Calculate
	## initialize things
	n = 800
	adfs = np.linspace(0.01,0.8,n)
	atom_weights = np.array([.05,1.,1.,1.,2.,2.])
	atom_types = np.array(['H','C','N','O','P','S'])
	weights = np.zeros(mol.natoms)
	for ati in range(atom_types.size):
		weights[mol.element == atom_types[ati]] = atom_weights[ati]
	probs = []


	###### Global
	lnev_global = np.zeros_like(adfs)
	for i in range(adfs.size):
		model = harp.models.density_atoms(grid, mol.xyz, weights, adfs[i], 5, .5)
		lnev_global[i] = harp.evidence.ln_evidence(model, density)
	lnev_global -= lnev_global.max()
	prob_global = 1./np.sum(np.exp(lnev_global[:,None]-lnev_global[None,:]),axis=0)/(adfs[1]-adfs[0])

	###### Individual
	## just loop through ADFs and ignore the blob calculation
	for chain in mol.unique_chains:
		## Extract current chain
		subchain = mol.get_chain(chain)
		if np.all(subchain.hetatom):
			continue

		## Loop over residues in current chain
		for i in tqdm.tqdm(range(subchain.unique_residues.size)):
			# print(chain,i)
			## Extract current residue
			resi = subchain.unique_residues[i]
			subresidue = subchain.get_residue(resi)

			prob,ln_ev = harp.bayes_model_select.bms_residue(grid,density,subresidue,adfs,np.array((1.,)),8., 5,0.5,atom_types,atom_weights)
			ln_ev = ln_ev[0,:adfs.size] ## remove the placeholder blob model and recalculate prob
			ind = ln_ev.argmax()
			ln_ev -= ln_ev.max()
			prob = 1./np.nansum(np.exp(ln_ev[:,None]-ln_ev[None,:]),axis=0)
			probs.append(prob)
	## collect info
	probs = np.array(probs)
	probs[np.bitwise_not(np.isfinite(probs))] = 0.
	probs /= (adfs[1]-adfs[0])
	rs = np.array([adfs[pp.argmax()] for pp in probs])




	#### Plots - 1
	fig,ax=plt.subplots(1)

	box = AnchoredOffsetbox(child=TextArea("N=%d"%(probs.shape[0])),loc="upper right", frameon=True)
	box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
	ax.add_artist(box)

	for i in range(probs.shape[0]):
		ax.plot(adfs-adfs[probs[i].argmax()],probs[i],color='k',alpha=.1,lw=1)

	ax.set_xlabel('$\sigma_{ADF} - \hat{\sigma}^{MAP}_{ADF}$    $(\AA)$')
	ax.set_ylabel('Posterior Probability')
	ax.set_xlim(-.15,.15)
	plt.title('Posterior widths for BMS of $\sigma_{ADF}$ (%s)'%(pdbid.upper()))
	plt.savefig('figures/rendered/byresidue_adfopt_posteriorwidth_%s.pdf'%(pdbid))
	plt.savefig('figures/rendered/byresidue_adfopt_posteriorwidth_%s.png'%(pdbid),dpi=300)
	plt.close()



	#### Plots - 2
	fig,ax=plt.subplots(1)

	# pensemble = np.sum(probs,axis=0)/np.sum(probs)
	ax.plot(adfs,probs.mean(0),color='tab:blue',lw=1.2,alpha=1.,label='Ensemble Average',zorder=2)
	ax.plot(adfs,prob_global,color='tab:orange',lw=1.2,alpha=1.,linestyle='-',label='Global',zorder=1)

	xlh,xmap,xmu,xstd,order = hpdi_post_all(adfs,probs,frac=.95)
	for i in range(len(probs)):
		prob = probs[order[i]]
		ax.fill_between(adfs,prob,color='white',edgecolor='k',linewidth=.8,alpha=.4,zorder=-1)
		# ax.plot(adfs,prob,color='k',lw=.8,alpha=.066,zorder=-1)


	ax.set_ylim(-3,probs.max()+3.)
	ax.set_xlim(0.2,.65)
	plt.title('Posteriors for BMS of $\sigma_{ADF}$ (%s)'%(pdbid.upper()))
	plt.xlabel('$\sigma_{ADF}$   $(\AA)$')
	plt.ylabel('Posterior Probability')
	plt.legend()
	plt.savefig('figures/rendered/byresidue_adfopt_posteriors_%s.pdf'%(pdbid))
	plt.savefig('figures/rendered/byresidue_adfopt_posteriors_%s.png'%(pdbid),dpi=300)
	plt.close()



	#### Plots - 3
	fig,ax=plt.subplots(1,3,figsize=(10,3))

	mmax = probs.max()*1.02
	mmin = probs.max()*.01
	xlh,xmap,xmu,xstd,order = hpdi_post_all(adfs,probs,frac=.95)
	xres = (np.linspace(0,1,order.size)/float(order.size)*(mmax-mmin)+mmin)[::-1]

	for i in range(order.size):
		ii = order[i]
		ax[0].fill_between(adfs,probs[ii],color='white',edgecolor='k',linewidth=.8,alpha=.4,zorder=1)
		ax[1].plot(xlh[ii],[xres[i],xres[i]],color='tab:blue',lw=.8,alpha=.6,zorder=2)

	ax[1].plot(xmap[order],xres,color='tab:blue',zorder=3)

	delta1 = xmu.max()-xmu.min()
	delta2 = xstd.max()-xstd.min()
	im = ax[2].hist2d(xmu[order],xstd[order],bins=20,cmin=1,range=[[xmu.min()-.05*delta1,xmu.max()+.05*delta1],[xstd.min()-.05*delta2,xstd.max()+.05*delta2]])[3]
	fig.colorbar(im,ax=ax[2])

	box = AnchoredOffsetbox(child=TextArea("N=%d"%(probs.shape[0])),loc="upper right", frameon=True)
	box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
	ax[0].add_artist(box)

	ax[0].set_ylim(0,mmax+.02)
	# ax[0].set_xlim(.2,.6)
	ax[0].set_xlabel(r'$\sigma_{ADF}$  $(\AA)$')
	ax[1].set_yticks(())
	ax[1].set_xlabel(r'MAP and 95% HPDI of $\sigma_{ADF}$')
	ax[2].set_xlabel(r'$\mathbb{E}[\sigma_{ADF}]$')
	ax[2].set_ylabel(r'$\sqrt{\mathbb{E}[(\sigma_{ADF}^2]-\mathbb{E}[(\sigma_{ADF}]^2}$')
	ax[0].set_ylabel('Posterior Probability Density')
	ax[1].set_ylabel('Residues')
	plt.suptitle('%s'%(pdbid.upper()))
	fig.tight_layout()
	plt.savefig('figures/rendered/byresidue_adfopt_distributions_%s.pdf'%(pdbid))
	plt.savefig('figures/rendered/byresidue_adfopt_distributions_%s.png'%(pdbid),dpi=300)
	plt.close()

	####################################################################################
	################################# Blobs ############################################
	####################################################################################
	blobs = np.linspace(.25,8.,800)
	# blobs = np.concatenate((blobs[:-1],np.linspace(blobs.max(),10.0,20)))
	# blobs = np.concatenate((blobs[:-1],np.linspace(blobs.max(),20.0,11)))
	# blobs = np.concatenate((blobs[:-1],np.linspace(blobs.max(),100.0,9)))
	probs = []

	for chain in mol.unique_chains:
		## Extract current chain
		subchain = mol.get_chain(chain)
		if np.all(subchain.hetatom):
			continue

		## Loop over residues in current chain
		for i in tqdm.tqdm(range(subchain.unique_residues.size)):
			# print(chain,i)
			## Extract current residue
			resi = subchain.unique_residues[i]
			subresidue = subchain.get_residue(resi)

			prob,ln_ev = harp.bayes_model_select.bms_residue(grid,density,subresidue,np.array((.1,)),blobs,8.,5,0.5,atom_types,atom_weights)
			ln_ev = ln_ev[0,1:] ## remove the placeholder adf model and recalculate prob
			ind = ln_ev.argmax()
			ln_ev -= ln_ev.max()
			prob = 1./np.nansum(np.exp(ln_ev[:,None]-ln_ev[None,:]),axis=0)
			probs.append(prob)
	## collect info
	probs = np.array(probs)
	probs[np.bitwise_not(np.isfinite(probs))] = 0.
	probs /= (blobs[1]-blobs[0])
	rs = np.array([blobs[pp.argmax()] for pp in probs])




	#### Plot - 4
	fig,ax=plt.subplots(1,3,figsize=(10,3))

	mmax = probs.max()*1.02
	mmin = probs.max()*.01
	xlh,xmap,xmu,xstd,order = hpdi_post_all(blobs,probs,frac=.95)
	xres = (np.linspace(0,1,order.size)/float(order.size)*(mmax-mmin)+mmin)[::-1]

	for i in range(order.size):
		ii = order[i]
		ax[0].fill_between(blobs,probs[ii],color='white',edgecolor='k',linewidth=.8,alpha=.4,zorder=1)
		ax[1].plot(xlh[ii],[xres[i],xres[i]],color='tab:blue',lw=.8,alpha=.6,zorder=2)
	ax[1].plot(xmap[order],xres,color='tab:blue',zorder=3)

	delta1 = xmu.max()-xmu.min()
	delta2 = xstd.max()-xstd.min()
	im = ax[2].hist2d(xmu[order],xstd[order],bins=20,cmin=1,range=[[xmu.min()-.05*delta1,xmu.max()+.05*delta1],[xstd.min()-.05*delta2,xstd.max()+.05*delta2]])[3]
	fig.colorbar(im,ax=ax[2])

	box = AnchoredOffsetbox(child=TextArea("N=%d"%(probs.shape[0])),loc="upper right", frameon=True)
	box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
	ax[0].add_artist(box)

	ax[0].set_ylim(0,mmax+.02)
	ax[0].set_xlim(.0,2.5)
	ax[0].set_xlabel(r'$\sigma_{blob}$  $(\AA)$')
	ax[1].set_yticks(())
	ax[1].set_xlabel(r'MAP and 95% HPDI of $\sigma_{blob}$')
	ax[2].set_xlabel(r'$\mathbb{E}[\sigma_{blob}]$')
	ax[2].set_ylabel(r'$\sqrt{\mathbb{E}[(\sigma_{blob}^2]-\mathbb{E}[(\sigma_{blob}]^2}$')
	ax[0].set_ylabel('Posterior Probability Density')
	ax[1].set_ylabel('Residues')
	plt.suptitle('%s'%(pdbid.upper()))

	fig.tight_layout()
	plt.savefig('figures/rendered/byresidue_blobopt_distributions_%s.pdf'%(pdbid))
	plt.savefig('figures/rendered/byresidue_blobopt_distributions_%s.png'%(pdbid),dpi=300)
	plt.close()
