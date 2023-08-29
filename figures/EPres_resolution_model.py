## 23

import common_figures as cf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

##### MLE linear modeling with model selection for switching point

def piecewise_fit(x,mu_as,mu_bs,tau_as,tau_bs,covs,cut):
	nsamples=1000
	rvs = np.zeros((x.size,nsamples,4))+np.nan
	for i in range(x.size):
		if not np.any(np.isnan(covs[i])):
			rvs[i] = np.random.multivariate_normal(np.array((mu_as[i],mu_bs[i],tau_as[i],tau_bs[i])),covs[i],size=nsamples)
	qa = np.exp(rvs[:,:,0])
	qb = np.exp(rvs[:,:,1])

	x1 = x[:cut]
	y1 = np.mean((qa/(qa+qb))[:cut],axis=1)
	s21 = np.var((qa/(qa+qb))[:cut],axis=1)
	x2 = x[cut:]
	y2 = np.mean((qa/(qa+qb))[cut:],axis=1)
	s22 = np.var((qa/(qa+qb))[cut:],axis=1)


	m1 = (np.sum(1./s21)*np.sum(x1*y1/s21)-np.sum(x1/s21)*np.sum(y1/s21))/(np.sum(1./s21)*np.sum(x1*x1/s21)-np.sum(x1/s21)*np.sum(x1/s21))
	b1 = (np.sum(y1/s21)-m1*np.sum(x1/s21))/(np.sum(1./s21))
	m2 = (np.sum(1./s22)*np.sum(x2*y2/s22)-np.sum(x2/s22)*np.sum(y2/s22))/(np.sum(1./s22)*np.sum(x2*x2/s22)-np.sum(x2/s22)*np.sum(x2/s22))
	b2 = (np.sum(y2/s22)-m2*np.sum(x2/s22))/(np.sum(1./s22))

	return m1,b1,m2,b2

def lnL(x,mu_as,mu_bs,tau_as,tau_bs,covs,cut,m1,b1,m2,b2):
	nsamples=1000
	rvs = np.zeros((x.size,nsamples,4))+np.nan
	for i in range(x.size):
		if not np.any(np.isnan(covs[i])):
			rvs[i] = np.random.multivariate_normal(np.array((mu_as[i],mu_bs[i],tau_as[i],tau_bs[i])),covs[i],size=nsamples)
	qa = np.exp(rvs[:,:,0])
	qb = np.exp(rvs[:,:,1])

	x1 = x[:cut]
	y1 = np.mean((qa/(qa+qb))[:cut],axis=1)
	s21 = np.var((qa/(qa+qb))[:cut],axis=1)
	x2 = x[cut:]
	y2 = np.mean((qa/(qa+qb))[cut:],axis=1)
	s22 = np.var((qa/(qa+qb))[cut:],axis=1)

	out  = np.sum(-.5*np.log(2.*np.pi*s21)) + np.sum(-.5/s21*(y1-(m1*x1+b1))**2.)
	out += np.sum(-.5*np.log(2.*np.pi*s22)) + np.sum(-.5/s22*(y2-(m2*x2+b2))**2.)
	return out


resolution = cf.load_resolution('./all_results.hdf5')
deposit_date = cf.load_depositdate('./all_results.hdf5')
P_res = cf.load_P_res('./all_results.hdf5')
print('loaded')


# for ycut in [2019,2018,2014,2000]:
# for ycut in [2000,2014,2018,2019]:
for ycut in [2018,]:
	res = np.arange(0,8.,.1)
	P_res_res = [[] for _ in range(res.size)]
	for i in range(len(deposit_date)):
		y,m,d = deposit_date[i].split('-')
		if int(y) >= ycut:
			pp = P_res[i].copy()
			pp[~np.isfinite(pp)] = 0.
			if resolution[i] > res[-1]:
				index = res.size-1
			else:
				index = np.argmax(res >= resolution[i])-1
			P_res_res[index].append(pp)
	print('sorted')

	ns = np.array([len(prri) for prri in P_res_res])
	keep = ns > 1
	resk = res[keep]
	P_res_resk = [P_res_res[i] for i in range(len(P_res_res)) if keep[i]]

	# mps,fig = cf.process_sets_indiv(resk,P_res_resk,width=.07,alpha0=1.,beta0=1.)
	ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_res_resk,'figures/models/hmodel_resolution_%d.hdf5'%(ycut),nres=20)


	alphas = np.exp(mu_as)
	betas = np.exp(mu_bs)

	bestl = -np.inf
	bestfits = []
	bestcut = 0
	for cut in range(2,len(resk)-2):
		m1,b1,m2,b2 = piecewise_fit(resk,mu_as,mu_bs,tau_as,tau_bs,covs,cut)
		l = lnL(resk,mu_as,mu_bs,tau_as,tau_bs,covs,cut,m1,b1,m2,b2)
		if l > bestl:
			bestl = l
			bestfits = [m1,b1,m2,b2]
			bestcut = cut

	m1,b1,m2,b2 = bestfits
	cut = bestcut
	res = np.linspace(0,8.2,1000)
	rcut = (b2-b1)/(m1-m2)
	ymodel = (m1*res+b1)*(res<rcut) + (m2*res+b2)*(res>=rcut)


	###### plotting

	fig,ax = cf.make_fig_set_ab(resk,mu_as,mu_bs,tau_as,tau_bs,covs)

	for aa in fig.axes:
		aa.set_xticks(np.arange(8+1))
		aa.set_xticklabels(np.arange(8+1))
		aa.set_xlim(0.,8.)
	ax['P'].set_xlabel('FSC Resolution ($\AA$)')
	ax['B'].set_xlabel('FSC Resolution ($\AA$)')
	# ax['A'].set_ylim(6e-2,2e1)
	ax['A'].set_ylim(6e-2,.5e1)
	ax['B'].set_ylim(.06,6.)
	
	ax['P'].set_ylabel(r'$\langle P \rangle$')
	
	


	ax['P'].plot(res,ymodel,'k',alpha=1.,zorder=+3,lw=1.5)

	ax['P'].plot([0,(.5-b1)/m1],[.5,.5],color='k',ls='--')
	ax['P'].plot([(.5-b1)/m1,(.5-b1)/m1],[-.02,.5],color='k',ls='--')

	anchored_text = AnchoredText(r'$y_{high}$ = %.3fx + %.2f'%(m1,b1) +'\n' + r'$y_{low }$ = %.3fx + %.2f'%(m2,b2) +'\n' + r'$\langle P\rangle (%.2f \AA)$ = 0.5'%((.5-b1)/m1), loc=1,prop={'ha':'right'})
	ax['P'].add_artist(anchored_text)


	fig.savefig('figures/rendered/EPres_resolution_model_%d-2022.pdf'%(ycut),dpi=600)
	fig.savefig('figures/rendered/EPres_resolution_model_%d-2022.png'%(ycut),dpi=600)

	plt.close()
	
	
	# break
