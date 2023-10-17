'''
Notes:
Install http://openbabel.org/wiki/Category:Installation
Mac -- brew install open-babel

'''

import os
import numpy as np
import matplotlib.pyplot as plt
import logging

aas = {
'Gly':'C(C(=O)O)N',
'Pro':'C1CC(NC1)C(=O)O',
'Ala':'CC(C(=O)O)N',
'Val':'CC(C)C(C(=O)O)N',
'Leu':'CC(C)CC(C(=O)O)N',
'Ile':'CCC(C)C(C(=O)O)N',
'Met':'CSCCC(C(=O)O)N',
'Cys':'C(C(C(=O)O)N)S',
'Phe':'C1=CC=C(C=C1)CC(C(=O)O)N',
'Tyr':'C1=CC(=CC=C1CC(C(=O)O)N)O',
'Trp':'C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N',
'His':'C1=C(NC=N1)CC(C(=O)O)N',
'Lys':'C(CCN)C[C@@H](C(=O)O)N',
'Arg':'C(CC(C(=O)O)N)CN=C(N)N',
'Gln':'C(CC(=O)N)C(C(=O)O)N',
'Asn':'C(C(C(=O)O)N)C(=O)N',
'Glu':'C(CC(=O)O)C(C(=O)O)N',
'Asp':'C(C(C(=O)O)N)C(=O)O',
'Ser':'C(C(C(=O)O)N)O',
'Thr':'CC(C(C(=O)O)N)O',
}

def run_obabel(aa_list,fdir):
	logging.debug('Running obabel')
	try:
		os.mkdir(fdir)
	except:
		pass
	for aaname in aa_list.keys():
		smiles = aa_list[aaname]
		logging.debug("obabel: %s, %s"%(aaname,smiles))
		cmd = 'obabel -:\"%s\" --gen2d -o mmcif -O %s/%s.cif'%(smiles, fdir, aaname)
		try:
			stream = os.popen(cmd)
			output = stream.read()
			logging.debug(output)
		except:
			pass

def get_xyz(fn):
	from blobornot import io
	mol = io.mmcif.load_mmcif_dict(fn)
	mas = mol['_atom_site']
	atoms = np.array([masi[1] for masi in mas[1]])
	heavy = atoms != 'H'
	xyz = np.array([[float(masi[i]) for masi in mas[1]] for i in [2,3,4]]).T
	return atoms[heavy],xyz[heavy]

def get_backbone(atoms,xyz):
	keepn = np.nonzero(atoms == 'N')[0]
	keepo = np.nonzero(atoms == 'O')[0]
	keepc = np.nonzero(atoms == 'C')[0]
	o_d = []
	o_i = []
	for i in keepn:
		for jj in keepc:
			for j in keepc:
				if j > jj:
					for k in keepo:
						for l in keepo:
							if l > k:
								r = np.linalg.norm(xyz[i]-xyz[j])++np.linalg.norm(xyz[j]-xyz[jj])+np.linalg.norm(xyz[j]-xyz[k])+np.linalg.norm(xyz[j]-xyz[l])
								o_d.append(r)
								o_i.append([i,jj,j,k,l])
	o_d = np.array(o_d)
	o_i = np.array(o_i)
	backbone_i = o_i[o_d.argmin()]
	return backbone_i

def align_backbone(atoms,xyz):
	backbone = get_backbone(atoms,xyz)
	xyz -= xyz[backbone[1]][None,:]
	nxyz = xyz[backbone[2]]
	theta = -np.arctan(nxyz[1]/nxyz[0])
	if nxyz[0] < 0:
		theta += np.pi
	newx = xyz[:,0]*np.cos(theta) - xyz[:,1]*np.sin(theta)
	newy = xyz[:,0]*np.sin(theta) + xyz[:,1]*np.cos(theta)
	xyz[:,0] = newx
	xyz[:,1] = -newy
	return xyz

def despine(aa):
	aa.spines['right'].set_visible(False)
	aa.spines['top'].set_visible(False)
	aa.yaxis.set_ticks_position('left')
	aa.xaxis.set_ticks_position('bottom')

colors = {
#JMol
'C':'#909090',
'N':'#3050F8',
'O':'#FF0D0D',
'S':'#FFFF30'
}

def plot_aa(atoms,xyz,ax,resname,blist=None):
	global clors
	if not blist is None:
		for bb in blist:
			bond(bb[0],bb[1],xyz,ax)
	ns = np.unique(atoms)
	patches = []
	for nsi in ns:
		# ax.plot(xyz[atoms==nsi,0],xyz[atoms==nsi,1],'o',color=colors[nsi],ms=4)
		for i in range(np.sum(atoms==nsi)):
			circle = plt.Circle((xyz[atoms==nsi,0][i],xyz[atoms==nsi,1][i]),radius=.4,fc=colors[nsi],ec='none',zorder=2)
			ax.add_artist(circle)
	ax.annotate(resname,xy=(.05,.8),xycoords='axes fraction')
	fix_axis(ax)

def fix_axis(ax):
	ax.set_xticks((),())
	ax.set_yticks((),())
	ax.set_xlim(-5,5)
	ax.set_ylim(-5,5)
	ax.set_aspect(1.)
	ax.axis('off')

def get_aa(aa,fdir):
	fname = os.path.join(fdir,aa+'.cif')
	atoms,xyz = get_xyz(fname)

	## this works for most of the amino acids
	xyz = align_backbone(atoms,xyz)

	com = xyz.mean(0)
	xyz -= com[None,:]
	return atoms,xyz

def bond(i,j,xyz,ax):
	ax.plot(xyz[[i,j],0],xyz[[i,j],1],color='k')
	# print(np.linalg.norm(xyz[i]-xyz[j]))

def makefig_show_aas(figdir,aadir):
	logging.debug('Cartoon figure: making all_aas')
	fig,ax = plt.subplots(4,5,figsize=(5.5,4.4))
	plt.subplots_adjust(hspace=.00,wspace=.00)
	pnum = 0
	for aa in aas.keys():
		atoms,xyz = get_aa(aa,aadir)

		i = pnum // 5
		j = pnum %5
		plot_aa(atoms,xyz,ax[i,j],aa)
		pnum += 1

	plt.savefig(os.path.join(figdir,'cartoon_all_aas.png'))
	plt.savefig(os.path.join(figdir,'cartoon_all_aas.pdf'))
	plt.close()

def makefig_cartoon_pnot(figdir,aadir):
	logging.debug('Cartoon figure: making pnot')
	from blobornot import density,models
	aas = ['Phe','Ala']
	blist = [
	[[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[3,6],[6,7],[7,8],[8,9],[8,10],[7,11]],
	[[0,1],[1,2],[2,3],[2,4],[1,5]]
	]

	for res in [.2,.5]:
		fig,ax = plt.subplots(2,5,figsize=(6,3),dpi=300)
		plt.subplots_adjust(hspace=.02,wspace=.02,left=.05,right=.95,bottom=.05,top=.85)
		for i in range(len(aas)):
			atoms,xyz = get_aa(aas[i],aadir)
			xyz *= 1.55
			for j in range(5):
				if j == 0:
					plot_aa(atoms,xyz,ax[i,j],aas[i],blist[i])
				com = xyz.mean(0)
				origin = np.zeros(3) - 5
				dxyz = np.zeros(3) + res
				nxyz = ((np.zeros(3)+5. - origin)//dxyz).astype('int')
				grid = density.gridclass(origin,dxyz,nxyz)
				x = np.linspace(origin[0],origin[0]+nxyz[0]*dxyz[0],nxyz[0])
				y = np.linspace(origin[1],origin[1]+nxyz[1]*dxyz[1],nxyz[1])
				cmap = plt.cm.plasma
				if j == 1:
					moldensity = models._render_model(grid.origin,grid.dxyz,grid.nxyz,xyz,.4,8,.0)
					title = r'$\mathrm{\sigma_{atom}=0.4\AA}$'
				elif j == 2:
					moldensity = models._render_model(grid.origin,grid.dxyz,grid.nxyz,xyz,.7,8,.0)
					title = r'$\mathrm{\sigma_{atom}=0.7\AA}$'
				elif j == 3:
					moldensity = models._render_model(grid.origin,grid.dxyz,grid.nxyz,com[None,:],1.5,8,.0)
					title = r'$\mathrm{\sigma_{blob}=1.5\AA}$'
				elif j == 4:
					moldensity = models._render_model(grid.origin,grid.dxyz,grid.nxyz,com[None,:],2.5,8,.0)
					title = r'$\mathrm{\sigma_{blob}=2.5\AA}$'

				if j > 0:
					# moldensity = moldensity[:,:,nxyz[2]//2]
					moldensity = moldensity.sum(2)
					moldensity /= moldensity.max()
					# moldensity = np.log(moldensity)
					if i == 0:
						ax[i,j].set_title(title)
					ax[i,j].imshow(moldensity.T,extent=((x[0],x[-1],y[0],y[-1])),origin='lower',cmap=cmap,vmin=0, vmax=moldensity.max())
					fix_axis(ax[i,j])

		if res == .2:
			rs = 'high'
		elif res == .5:
			rs = 'low'
		plt.savefig(os.path.join(figdir,'cartoon_pnot_%s.pdf'%(rs)))
		plt.savefig(os.path.join(figdir,'cartoon_pnot_%s.png'%(rs)))
		plt.close()
	# plt.show()


def makefig_cartoon_shape(figdir,aadir):
	logging.debug('Cartoon figure: making ms_shape')
	from blobornot import density,models,evidence
	fig,ax = plt.subplots(2,2,figsize=(5,4))
	fig.subplots_adjust(left=.05,right=.95,top=.9,bottom=.1,hspace=.2,wspace=.25)
	aa = 'Phe'
	atoms,xyz = get_aa(aa,aadir)

	lim = 2.
	origin = np.zeros(3) - lim
	dxyz = np.zeros(3) + .1
	nxyz = ((np.zeros(3)+lim - origin)//dxyz).astype('int') + 1
	grid = density.gridclass(origin,dxyz,nxyz)

	x = np.linspace(origin[0],origin[0]+nxyz[0]*dxyz[0],nxyz[0])
	y = np.linspace(origin[1],origin[1]+nxyz[1]*dxyz[1],nxyz[1])

	xyz = xyz[:6]

	com = xyz.mean(0)
	xyz -= com[None,:]


	np.random.seed(666)
	####
	moldensity = models._render_model(grid.origin,grid.dxyz,grid.nxyz,xyz,.2,8,.0)
	for i in range(20):
		theta = np.deg2rad(np.random.normal()*9.)
		# theta = np.deg2rad(i*.4)
		xyz2 = xyz.copy()
		bb = np.arange(xyz.shape[0])<=6
		xyz2[bb,0] = xyz[bb,0]*np.cos(theta) - xyz[bb,1]*np.sin(theta)
		xyz2[bb,1] = xyz[bb,0]*np.sin(theta) + xyz[bb,1]*np.cos(theta)
		# xyz2 += com[None,:]
		# xyz2[:,:2] += np.random.normal(size=2)*.25
		xyz2[:,:2] += np.random.normal(size=(xyz.shape[0],2))*.238
		moldensity += models._render_model(grid.origin,grid.dxyz,grid.nxyz,xyz2,.2,8,.0)*(1.+np.random.rand()*.2)
		# plot_aa(atoms,xyz2,ax,aa,blist[0])

	moldensity = moldensity.sum(2)
	moldensity/=moldensity.sum()
	moldensity *= 1000000
	moldensity = np.random.normal(size=moldensity.shape)*10. + moldensity
	d1 = moldensity.copy()

	####
	moldensity = models._render_model(grid.origin,grid.dxyz,grid.nxyz,xyz,.3,8,.0)
	moldensity = moldensity.sum(2)
	moldensity/=moldensity.sum()
	d2 = moldensity.copy()
	# moldensity*=10000
	# moldensity = np.random.poisson(moldensity)


	####
	moldensity = models._render_model(grid.origin,grid.dxyz,grid.nxyz,xyz,.2,8,.0)
	for i in range(60):
		# theta = np.deg2rad(np.random.normal()*17.5)
		theta = np.deg2rad(i)
		xyz2 = xyz.copy()
		bb = np.arange(xyz.shape[0])<=6
		xyz2[bb,0] = xyz[bb,0]*np.cos(theta) - xyz[bb,1]*np.sin(theta)
		xyz2[bb,1] = xyz[bb,0]*np.sin(theta) + xyz[bb,1]*np.cos(theta)
		# xyz2 += com[None,:]
		# xyz2[:,:2] += np.random.normal(size=2)*.25
		moldensity += models._render_model(grid.origin,grid.dxyz,grid.nxyz,xyz2,.3,8,.0)
		# plot_aa(atoms,xyz2,ax,aa,blist[0])

	moldensity = moldensity.sum(2)
	moldensity/=moldensity.sum()
	d3 = moldensity.copy()


	#####
	cmap = plt.cm.Reds
	ax[1,0].imshow(d1.T,extent=((x[0],x[-1],y[0],y[-1])),origin='lower',cmap=cmap,vmin=0, vmax=d1.max())
	cmap = plt.cm.Greens
	ax[0,0].imshow(d2.T,extent=((x[0],x[-1],y[0],y[-1])),origin='lower',cmap=cmap,vmin=0, vmax=d2.max())
	cmap = plt.cm.Blues
	ax[0,1].imshow(d3.T,extent=((x[0],x[-1],y[0],y[-1])),origin='lower',cmap=cmap,vmin=0, vmax=d3.max())
	ax[1,0].set_title('Observed')
	ax[0,0].set_title('Hexagon')
	ax[0,1].set_title('Circle')

	####
	lnev2 = evidence.ln_evidence(d1,d2)
	lnev3 = evidence.ln_evidence(d1,d3)
	r = np.exp(lnev3+np.log(.5)-lnev2-np.log(.5))
	p2 = 1./(1.+r)
	# print((lnev2,lnev3,r,p2))

	# ax[1,1].set_title('Model Selection')
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	divider = make_axes_locatable(ax[1,1])
	ax2 = divider.append_axes("right", size="50%", pad="60%")
	ax[1,1].plot([1,],[lnev2,],marker="o", linestyle="",color='tab:green')
	ax[1,1].plot([2,],[lnev3,],marker="o", linestyle="",color='tab:blue')
	ax[1,1].set_xticks([1,2])
	ax[1,1].set_xlim(.75,2.25)
	delta = lnev2-lnev3
	ax[1,1].set_ylim(10900,10904)
	ax[1,1].set_yticks(np.arange(5)+10900)
	ax[1,1].set_yticklabels(np.arange(5)+10900)
	ax[1,1].set_xticklabels(['Hexagon','Circle'])
	ax2.bar([1,],[1,],color='tab:blue')
	ax2.bar([1,],[p2,],color='tab:green')
	ax2.set_xticks((),())
	ax2.set_ylabel(r'$\mathrm{P_{Hexagon}}$')
	ax[1,1].set_ylabel(r'$\mathrm{\ln\left(Evidence\right)}$')
	ax2.set_ylim(0,1)

	for aa in [ax[1,1],ax2]:
		despine(aa)


	####
	for i in range(2):
		for j in range(2):
			if not ( i == 1 and j == 1):
				fix_axis(ax[i,j])
				ax[i,j].set_xlim(-2,2)
				ax[i,j].set_ylim(-2,2)

	# fig.tight_layout()
	plt.savefig(os.path.join(figdir,'cartoon_ms_shape.pdf'))
	plt.savefig(os.path.join(figdir,'cartoon_ms_shape.png'))
	plt.close()
	# plt.show()

def makefig_cartoon_calc(figdir):
	logging.debug('Cartoon figure: making ms_calc_ev')
	np.random.seed(666)
	plt.rc('font', size=12)
	plt.rc('axes', labelsize=14)

	lme1 = np.array([7.,4.,5.,3.5,2.5,1.5,1,.5])
	lme2 = np.array([2.,1.,1.5,2.5,4.5,6.,4.5,3.])
	lme1 += np.random.normal(size=lme1.size)*.25
	lme2 += np.random.normal(size=lme2.size)*.25

	p1 = 1./np.sum(np.exp(lme1 - lme1[0]))
	p2 = 1./np.sum(np.exp(lme2 - lme2[0]))

	x = np.arange(lme1.size)

	fig,ax = plt.subplots(1,figsize=(3.75,2.5))
	fig.subplots_adjust(bottom=.2,right=.9,left=.1)
	ax.plot(x,lme1,color='tab:green',marker='o')
	ax.plot(x,lme2,color='tab:blue',marker='o')

	pmin = 1.-.05
	pmax = 7. + .1
	offset = -.85
	height=1.
	# p = plt.Polygon(np.array(((pmin,offset),(pmax,offset),(pmax,offset-height))),closed=True,clip_on=False,fc='none',ec='black')
	p = plt.Polygon(np.array(((pmin,offset),(pmax,offset),(pmax,offset-height),(pmin,offset-.1))),closed=True,clip_on=False,fc='lightgrey',ec='black')

	ax.annotate(r'$\sigma_{\mathrm{blob}}$',xy=(pmax+.2,offset - height*.8), annotation_clip=False)
	ax.legend(('A','B'),loc=1,ncol=2,title='Residue',bbox_to_anchor=(1.12, 1.16),edgecolor='k')

	xtl = ['',]*x.size
	xtl[0] = 'Not'

	ax.set_xlabel('Model')
	ax.set_ylim(-.25,8.25)
	ax.set_xticks(x)
	ax.set_xticklabels(xtl)
	despine(ax)
	ax.set_yticks((),())
	ax.set_ylabel(r'$\mathrm{\ln(Evidence)}$')
	ax.add_artist(p)

	plt.savefig(os.path.join(figdir,'cartoon_ms_calc_ev.pdf'))
	plt.savefig(os.path.join(figdir,'cartoon_ms_calc_ev.png'))
	plt.close()

	fig,ax = plt.subplots(1,figsize=(1.75,2.5))
	fig.subplots_adjust(bottom=.2,right=.98,left=.375)
	ax.bar([1,],[p1,],color='tab:green')
	ax.bar([2,],[p2,],color='tab:blue')

	ax.set_xticks((1,2))
	ax.set_xticklabels(('A','B'))
	ax.set_xlabel('Residue')
	ax.set_ylabel(r'$\mathrm{P_{not}}$')
	ax.set_ylim(0,1)
	despine(ax)

	plt.savefig(os.path.join(figdir,'cartoon_ms_calc_p.pdf'))
	plt.savefig(os.path.join(figdir,'cartoon_ms_calc_p.png'))
	plt.close()


def make_figures():
	logging.debug('Running Cartoon Figures')

	dirs = {'aafigs':'./figures/temp','figs':'./figures/rendered'}

	run_obabel(aas,dirs['aafigs'])

	makefig_show_aas(dirs['figs'],dirs['aafigs'])
	makefig_cartoon_pnot(dirs['figs'],dirs['aafigs'])
	makefig_cartoon_shape(dirs['figs'],dirs['aafigs'])
	makefig_cartoon_calc(dirs['figs'])
	for f in os.listdir(dirs['aafigs']):
		os.remove(os.path.join(dirs['aafigs'],f))
	os.rmdir(dirs['aafigs'])


if __name__ == '__main__':
	make_figures()
