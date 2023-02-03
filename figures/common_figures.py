## 23

import h5py as h
import numpy as np
from scipy.special import betaincinv
import matplotlib.pyplot as plt
# import plotstyle
import common_infer_full_2_lognormal_faster as infer_full_2_lognormal

unique_residues_reg = ['A','ALA','ARG','ASN','ASP','C','CYS','DA','DC','DG','DT','G','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','U','VAL']

def get_pdbs(fname):
	with h.File(fname,'r') as f:
		pdbids = list(f.keys())
	return pdbids

def load_pdbids(fname):
	return np.array(get_pdbs(fname))

def load_chains(fname):
	with h.File(fname,'r') as f:
		pdbids = get_pdbs(fname)
		npdbs = len(pdbids)

		chains = []
		for i in range(npdbs):
			pdbid = pdbids[i]
			chains.append(f[pdbid]['chains'][:].astype('U'))
	return chains

def load_residues(fname):
	with h.File(fname,'r') as f:
		pdbids = get_pdbs(fname)
		npdbs = len(pdbids)

		residues = []
		for i in range(npdbs):
			pdbid = pdbids[i]
			residues.append(f[pdbid]['resnames'][:].astype('U'))
	return residues

def load_resids(fname):
	with h.File(fname,'r') as f:
		pdbids = get_pdbs(fname)
		npdbs = len(pdbids)

		residues = []
		for i in range(npdbs):
			pdbid = pdbids[i]
			residues.append(f[pdbid]['resids'][:].astype('U'))
	return residues

def load_P_res(fname):
	with h.File(fname,'r') as f:
		pdbids = get_pdbs(fname)
		npdbs = len(pdbids)

		P_res = []
		alpha = np.zeros(npdbs)
		beta = np.zeros(npdbs)
		beta_P_res = np.zeros((3,npdbs))

		for i in range(npdbs):
			pdbid = pdbids[i]

			## get P_res values and sanitize
			pp = f[pdbid]['probs'][:]
			pp[~np.isfinite(pp)] = np.nan
			P_res.append(pp.copy())

	return P_res

def _load_string(meta_string,fname):
	with h.File(fname,'r') as f:
		pdbids = get_pdbs(fname)
		npdbs = len(pdbids)
		sarray = []
		for i in range(npdbs):
			pdbid = pdbids[i]
			sarray.append(f[pdbid].attrs[meta_string])
	return sarray

def _load_int(meta_string,fname):
	with h.File(fname,'r') as f:
		pdbids = get_pdbs(fname)
		npdbs = len(pdbids)
		out = np.zeros(npdbs,dtype='int')
		for i in range(npdbs):
			pdbid = pdbids[i]
			try:
				out[i] = int(f[pdbid].attrs[meta_string])
			except:
				out[i] = np.nan
	return out

def _load_float(meta_string,fname):
	with h.File(fname,'r') as f:
		pdbids = get_pdbs(fname)
		npdbs = len(pdbids)
		out = np.zeros(npdbs,dtype='double')
		for i in range(npdbs):
			pdbid = pdbids[i]
			try:
				out[i] = float(f[pdbid].attrs[meta_string])
			except:
				out[i] = np.nan
	return out

def load_alignment(fname):
	align = _load_float('alignment',fname)
	return align
def load_depositdate(fname):
	## - delimited strings
	return _load_string('deposit date','./all_results.hdf5')

def load_year(fname):
	year = _load_int('deposit year',fname)
	return year
def load_resolution(fname):
	resolution = _load_float('resolution',fname)
	return resolution
def load_cs(fname):
	cs = _load_float('cs',fname)
	return cs
def load_dose(fname):
	dose = _load_float('dose',fname)
	return dose
def load_voltage(fname):
	voltage = _load_float('voltage',fname)
	return voltage
def load_exposure(fname):
	exposure = _load_float('exposure',fname)
	return exposure
def load_numparticles(fname):
	num = _log_int('num particles',fname)
	return num
def load_numimgs(fname):
	num = _log_int('num images',fname)
	return num
def load_formula_weight(fname):
	soft = _load_float('formula weight',fname)
	return soft

def load_molwt(fname):
	molwt = _load_float('molwt',fname)
	units = _load_string('molwt units',fname)
	k = np.unique(units)
	mw = np.zeros_like(molwt)
	for i in range(molwt.size):
		if units[i].lower().startswith('mega'):
			mw[i] = molwt[i]*1e6
		elif units[i].lower().startswith('kilo'):
			mw[i] = molwt[i]*1e3
		else:
			mw[i] = np.nan
	return mw

def load_microscope(fname):
	microscope = _load_string('microscope',fname)
	return microscope

def load_camera(fname):
	camera = _load_string('camera',fname)
	for i in range(len(camera)):
		if camera[i][-1]==')':
			camera[i] = camera[i][:camera[i].index('(')]
		elif camera[i] in ['.','?']:
			camera[i] = 'missing entry'
	return camera

def load_software_recon(fname):
	soft = _load_string('software name - reconstruction',fname)
	for i in range(len(soft)):
		if soft[i] == '?':
			soft[i] = 'missing entry'
	return soft

def load_journal(fname):
	journal = _load_string('citation journal',fname)
	jdict = {
		'cell':'Cell',
		'elife':'eLife',
		'natcommun':'Nat. Comm.',
		'natstructmolbiol':'NSMB',
		'nature':'Nature',
		'procnatlacadsciusa':'PNAS',
		'science':'Science',
		'structure':'Structure',
		'tobepublished':'To be pub.'
	}
	for i in range(len(journal)):
		j = journal[i]
		jj = j.lower().replace('.',' ').replace(' ','')
		if jj in jdict:
			journal[i] = jdict[jj]
		else:
			journal[i] = 'other'
	return journal



# np_keys = ['citation year','cs','deposit year','dose','exposure','formula weight','humidity','mag','molwt','num frames','num imgs','num particles','num particles selected','pH','resolution','voltage']
#
# string_keys = ['camera','citation journal','cryogen','deposit date','dna','method','microscope','organism','organism id','pdb keywords','protein','resolution method','rna','software name - classification','software name - ctf correction','software name - final euler assignment','software name - image acquisition','software name - initial euler assignment','software name - model fitting','software name - model refinement','software name - particle selection','software name - reconstruction','software version - classification','software version - ctf correction','software version - final euler assignment','software version - image acquisition','software version - initial euler assignment','software version - model fitting','software version - model refinement','software version - particle selection','software version - reconstruction','status']



def diag_swarm(d,wrap=300,width=.45):
	x = d.copy()
	x.sort()
	y = np.arange(x.size).astype('double')%wrap
	y/=float(wrap-1)
	y*=width
	y-=width/2.
	cut = 6.
	y *= np.sin((np.arange(x.size)+y.size/cut)/((cut+2.)/cut*y.size)*np.pi)
	return y,x
def zigzag_swarm(d,wrap=300,width=.45):
	x = d.copy()
	x.sort()
	y = np.arange(x.size).astype('double')%wrap
	y/=float(wrap-1)
	y*=width
	y-=width/2.
	y *= np.sign(np.cos(np.arange(x.size)//wrap*np.pi))
	y *= np.sin((np.arange(x.size)+y.size/4.)/(1.5*y.size)*np.pi)
	return y,x
def sine_swarm(d,period=100.,width=.45):
	x = d.copy()
	x.sort()
	y = np.arange(x.size).astype('double')
	y = np.sin(y/period*np.pi) * np.sin((y+y.size/4.)/(1.5*y.size)*np.pi)

	y*=width/2.
	return y,x


def swarmbox(d,ind,ax,color=plt.cm.tab20b(0),wrap=40,width=.8):
	swarm = diag_swarm(d,wrap=wrap,width=width*.75)
	# swarm = zigzag_swarm(d,wrap=wrap,width=width*.75)
	# swarm = sine_swarm(d,period=wrap,width=width-.2)

	ax.plot(swarm[0]+ind,swarm[1],'.',color='k',alpha=.8,ms=1.)

	if d.size>3:
		bp = ax.boxplot(d,positions=[ind],zorder=2,showfliers=False,widths=[width],notch=False,patch_artist=True,medianprops={'color':'k','linewidth':1.5,'alpha':1.},whiskerprops={'color':'k','linewidth':1.},whis=0,capprops={'color':'k','linewidth':1})
		bp['boxes'][0].set_facecolor([color[0],color[1],color[2],0.6])
		bp['boxes'][0].set_edgecolor('k')


def make_fig_set_p(xs,ps,yscale='logit',width=.7):
	fig1,ax1 =plt.subplots(1,figsize=(6,8))
	for i in range(len(ps)):
		swarmbox(ps[i],xs[i],ax=ax1,color=plt.cm.tab20b(i%20),wrap=25,width=width)

	# ax1.set_xticks(np.arange(len(xs)))
	# ax1.set_xticklabels(xs,rotation=90)
	ax1.set_xlim(np.min(xs)-.5,np.max(xs)+.5)

	ymin = np.min([np.min(psi) for psi in ps if len(psi)>0])
	ymax = np.max([np.max(psi) for psi in ps if len(psi)>0])
	if yscale == 'logit':
		ax1.set_yscale('logit')
		ymin = np.log(ymin/(1.-ymin))
		ymax = np.log(ymax/(1.-ymax))
	delta = ymax-ymin
	ymin -= delta*.02
	ymax += delta*.02
	if yscale=='logit':
		ymin = np.exp(ymin)/(1.+np.exp(ymin))
		ymax = np.exp(ymax)/(1.+np.exp(ymax))
	ax1.set_ylim(ymin,ymax)
	# ax1.set_ylabel(r'$E[P_{res}]$')
	ax1.set_ylabel(r'$\langle P_{res} \rangle$')

	# ax1.set_xticklabels(['(n=%d) %s'%(np.size(ps[i]),str(xs[i])) for i in range(len(ps))])
	ax1.grid(axis='x')
	return fig1,ax1

def hpdi(samples,frac=.95):
	## highest posterior density interval -- from samples
	x = np.sort(samples)
	imin = 0
	imax = x.size-1
	Ntarget = int((frac*x.size)//1)
	if Ntarget < 2:
		raise Exception('idk what to do')
	while imax-imin+1 > Ntarget:
		delta_low = x[imax] - x[imin+1]
		delta_high = x[imax-1] - x[imin]
		if delta_low <= delta_high:
			imin += 1
		else:
			imax -= 1
	return np.array((x[imin],x[imax]))


def make_fig_set_ab(xs,mu_as,mu_bs,tau_as,tau_bs,covs):
	fig2 = plt.figure(constrained_layout=True,figsize=(8,3.5))
	ax2 = fig2.subplot_mosaic("PA\nPB",sharex=True)

	nsamples=1000
	rvs = np.zeros((xs.size,nsamples,2))+np.nan
	for i in range(xs.size):
		if not np.any(np.isnan(covs[i])):
			rvs[i] = np.random.multivariate_normal(np.array((mu_as[i],mu_bs[i])),covs[i,:2,:2],size=nsamples)
	qa = np.exp(rvs[:,:,0])
	qb = np.exp(rvs[:,:,1])
	qp = qa/(qa+qb)
	alphas = np.exp(mu_as)
	betas = np.exp(mu_bs)
	ps = alphas/(alphas+betas)

	yerr = np.zeros((2,xs.size))
	for i in range(xs.size):
		yerr[:,i] = hpdi(qp[i],.95)
	# ax2['P'].errorbar(xs,ps,yerr=yerr,marker='o',color='k')
	ax2['P'].plot(xs,ps,marker='.',color='black',lw=.75,markersize=5)
	ax2['P'].fill_between(xs,yerr[0],yerr[1],color='black',alpha=.2,zorder=2,edgecolor='black')
	ax2['P'].set_ylim(-.02,1.02)

	yerr = np.zeros((2,xs.size))
	for i in range(xs.size):
		yerr[:,i] = hpdi(qa[i],.95)
	# ax2['A'].errorbar(xs,alphas,yerr=yerr,marker='o',color='b',label=r'$\langle\alpha\rangle$')
	ax2['A'].fill_between(xs,yerr[0],yerr[1],color='b',alpha=.2,zorder=2,edgecolor='darkblue')
	ax2['A'].plot(xs,alphas,marker='.',color='b',label=r'$\langle\alpha\rangle$',lw=.75,markersize=5)
	yerr = np.zeros((2,xs.size))
	for i in range(xs.size):
		yerr[:,i] = hpdi(qb[i],.95)
	# ax2['B'].errorbar(xs,betas,yerr=yerr,marker='s',color='r',label=r'$\langle\beta\rangle$')
	ax2['B'].fill_between(xs,yerr[0],yerr[1],color='r',alpha=.2,zorder=2,edgecolor='darkred')
	ax2['B'].plot(xs,betas,marker='.',color='r',label=r'$\langle\beta\rangle$',lw=.75,markersize=5)

	ax2['B'].set_yscale('log')
	ax2['A'].set_yscale('log')
	ax2['B'].set_ylabel(r'$\langle\beta\rangle$')
	ax2['A'].set_ylabel(r'$\langle\alpha\rangle$')
	# ax2[1].legend()#loc=1)
	#
	# ax2[2].errorbar(xs,np.exp(1./np.sqrt(tau_bs)),yerr=qsb.std(1),marker='s',color='r',label=r'$Std(\beta)$')
	# ax2[2].errorbar(xs,np.exp(np.sqrt(1./tau_as)),yerr=qsa.std(1),marker='o',color='b',label=r'$Std(\alpha)$')
	# ax2[2].set_yscale('log')
	# ax2[2].legend()#loc=4)
	ax2['P'].set_ylabel(r'$\langle\langle P_{res} \rangle\rangle$')

	ax2['A'].set_xticklabels(())
	fig2.subplots_adjust(left=.08,right=.99,top=.98,bottom=.15,wspace=.21,hspace=.05)

	return fig2,ax2

process_sets_hierarchical = infer_full_2_lognormal.process_sets_hierarchical
process_sets_indiv = infer_full_2_lognormal.process_sets_indiv
