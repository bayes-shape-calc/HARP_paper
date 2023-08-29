import common_figures as cf
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt

def langevin_labels(labels,x_data,y_data,nsteps=1000,k0=10.,epsilon=1e-12,sigma=.1,dt=.05,l0=.15,kbT=10.,gamma=1.):
	#
	# nsteps = 1000
	# k0 = 10.
	# # k1 = 1.
	# epsilon = 1e-12
	# sigma = .1
	# dt = .05
	# l0 = .15
	# # l1 = .3
	# kbT = 10.
	# gamma = 1.
	#
	
	n = x_data.size

	## initialize w a clock
	com = np.array((np.sqrt(np.mean(x_data**2.)),np.sqrt(np.mean(y_data**2.))))
	rad = np.sqrt(np.mean((x_data-com[0])**2.+(y_data-com[1])**2.))*5

	theta = np.linspace(0,2*np.pi,2*n+1)[:-1]
	x_pos = com[0]+rad*np.cos(theta)
	y_pos = com[1]+rad*np.sin(theta)

	x_label = np.zeros_like(x_data)
	y_label = np.zeros_like(y_data)
	left = [i for i in range(theta.size)]
	for i in range(n):	
		rs = np.sqrt((x_data[i]-x_pos[left])**2.+(y_data[i]-y_pos[left])**2.)
		rind = rs.argmin()
		lind = left[rind]
		# print(i,rind,lind,rs)
		x_label[i] = x_pos[lind]
		y_label[i] = y_pos[lind]
		left.pop(rind)
	# x_label = x_data + np.random.normal(size=n)*.1
	# y_label = y_data + np.random.normal(size=n)*.1







	#### initialize
	mass = np.ones(n)
	pre = np.sqrt(2*mass*gamma*kbT)

	x_label0 = x_label.copy()
	y_label0 = y_label.copy()
	x_label1 = np.zeros_like(x_label0)
	y_label1 = np.zeros_like(y_label0)

	r2 = (x_label0-x_data)**2.+(y_label0-y_data)**2.
	r = np.sqrt(r2)
	## x
	vx = np.random.rand(n)*.01
	dux = k0*(r-l0)/r*(x_label0-x_data)
	accelx = (-dux-gamma*mass*vx+pre*np.random.normal(size=mass.size))/mass
	x_label1 = x_label0+vx*dt+.5*accelx*dt*dt
	## y
	vy = np.random.rand(n)*.01
	duy = k0*(r-l0)/r*(y_label0-y_data)
	accely = (-duy-gamma*mass*vy+pre*np.random.normal(size=mass.size))/mass
	y_label1 = y_label0+vy*dt+.5*accely*dt*dt



	record = np.zeros((nsteps,n,2))

	#### steps >1
	for step in range(nsteps):
		r2 = (x_label1-x_data)**2.+(y_label1-y_data)**2.
		r = np.sqrt(r2)
	
		vx = x_label1-x_label0
		vy = y_label1-y_label0
	
		## x,y
		dux = k0*(r-l0)/r*(x_label1-x_data)
		duy = k0*(r-l0)/r*(y_label1-y_data)
		for i in range(n):
			for j in range(n):
				if i != j:
					rij2 = (x_label1[i]-x_label1[j])**2.+(y_label1[i]-y_label1[j])**2.
					rij = np.sqrt(rij2)
					# dux[i] += k1*(rij-l1)/rij*(x_label1[i]-x_label1[j])
					# duy[i] += k1*(rij-l1)/rij*(y_label1[i]-y_label1[j])
					# duxi = np.min((1.,4*epsilon*(-12.*rij2**(-7.)+6*rij2**(-4.))*(x_label1[i]-x_label1[j])))
					# duyi = np.min((1.,4*epsilon*(-12.*rij2**(-7.)+6*rij2**(-4.))*(y_label1[i]-y_label1[j])))
					duxi = np.min((1.,4*epsilon*(-12.*rij2**(-7.))*(x_label1[i]-x_label1[j])))
					duyi = np.min((1.,4*epsilon*(-12.*rij2**(-7.))*(y_label1[i]-y_label1[j])))

					dux[i] += duxi
					duy[i] += duyi
		for i in range(n):
			for j in range(n):
				if i != j:
					rij2 = (x_label1[i]-x_data[j])**2.+(y_label1[i]-y_data[j])**2.
					rij = np.sqrt(rij2)
					# dux[i] += k1*(rij-l1)/rij*(x_label1[i]-x_label1[j])
					# duy[i] += k1*(rij-l1)/rij*(y_label1[i]-y_label1[j])
					duxi = np.min((1.,4*epsilon*(-12.*rij2**(-7.)+6*rij2**(-4.))*(x_label1[i]-x_data[j])))
					duyi = np.min((1.,4*epsilon*(-12.*rij2**(-7.)+6*rij2**(-4.))*(y_label1[i]-y_data[j])))
					dux[i] += duxi
					duy[i] += duyi
				
				
		accelx = (-dux-gamma*mass*vx+pre*np.random.normal(size=mass.size))/mass
		accely = (-duy-gamma*mass*vy+pre*np.random.normal(size=mass.size))/mass
		# print(vx.max(),vy.max(),accelx.max(),accely.max())
	
		jumpx = vx*dt + .5*accelx*dt*dt
		jumpy = vy*dt + .5*accely*dt*dt
		jumpx[np.abs(jumpx) > .1] = np.sign(jumpx[np.abs(jumpx)>.1]) * .1
		jumpy[np.abs(jumpy) > .1] = np.sign(jumpy[np.abs(jumpy)>.1]) * .1
	
		x_label2 = x_label1 + jumpx
		y_label2 = y_label1 + jumpy

	
		## propagate
		record[step,:,0] = x_label2.copy()
		record[step,:,1] = y_label2.copy()
	
		x_label0 = x_label1.copy()
		x_label1 = x_label2.copy()
		y_label0 = y_label1.copy()
		y_label1 = y_label2.copy()
		

	x_label = x_label2.copy()
	y_label = y_label2.copy()

	return x_label,y_label,record


deposit_date = cf.load_depositdate('./all_results.hdf5')
P_res = cf.load_P_res('./all_results.hdf5')
residues = cf.load_residues('./all_results.hdf5')
resolution = cf.load_resolution('./all_results.hdf5')
print('loaded')
for i in range(len(residues)):
	residues[i] = residues[i].astype('U')

## Define Residue Groups
all_residues = np.unique(np.concatenate(residues))
standard_residues = ['A','ALA','ARG','ASN','ASP','C','CYS','DA','DC','DG','DT','G','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','U','VAL']
res_dna = ['DA','DC','DG','DT']
res_rna = ['A','C','G','U']
res_aa = [r for r in standard_residues if (not r in res_dna) and (not r in res_rna)]
res_ns = [r for r in all_residues if not r in standard_residues] ## ns=non-standard

## x coordinates for plots
x_dna = np.arange(4)
x_rna = 4+np.arange(4)
x_aa = 8+np.arange(len(res_aa))
xs = np.concatenate([x_dna,x_rna,x_aa])
x_ns = np.arange(len(res_ns))


initiates_aa = np.array([
[-1.667208, -1.519080, 1.249326, 3.570052], ### 966.426 
[-1.111805, -2.416486, 0.267453, 14.801143], ### 1320.389 
[-1.818346, -2.074335, 0.961617, 6.228729], ### 867.790 
[-1.871158, -1.753872, 1.872073, 4.240304], ### 752.262 
[-1.812251, -1.568608, 1.028937, 3.275973], ### 229.065 
[-1.641930, -2.272378, 0.504857, 10.560398], ### 817.104 
[-1.776310, -2.061064, 0.828518, 6.305567], ### 1010.913 
[-1.812733, -1.634535, 1.252082, 4.433449], ### 951.018 
[-1.461690, -2.353377, 0.170860, 18.137807], ### 484.815 
[-1.186715, -1.994174, 0.405747, 5.732653], ### 870.632 
[-1.054426, -2.124767, 0.314998, 6.255806], ### 1600.872 
[-1.417542, -2.344607, 0.341483, 12.080984], ### 1290.212 
[-1.223276, -2.139754, 0.280910, 7.800832], ### 399.883 
[-0.705159, -2.316938, 0.136702, 8.828582], ### 976.213 
[-1.741547, -1.909842, 0.806894, 3.050437], ### 695.187 
[-1.832606, -1.587168, 2.216017, 3.596666], ### 874.842 
[-1.635222, -1.726448, 0.743041, 3.954489], ### 813.825 
[-1.010496, -2.458828, 0.193184, 9502.021584], ### 371.564 
[-0.675695, -2.370495, 0.115421, 9.182216], ### 878.291 
[-1.360419, -1.782774, 0.570507, 4.113113], ### 913.048 
])
# initiates_aa = None

initiates_dna = np.array([
[-2.324427, -2.611187, 10.128654, 9550.234284], ### 246.647 
[-2.408822, -2.612963, 6742.804414, 9493.776151], ### 261.564 
[-2.392064, -2.597642, 9.732065, 9980.295407], ### 261.856 
[-2.443419, -2.583839, 5179.206167, 9087.417711], ### 234.450 
	
])

initiates_rna = np.array([
[-1.981343, -2.546032, 5.119177, 25.958471], ### 4166.456 
[-2.033606, -2.518322, 7.034091, 18.071054], ### 3735.469 
[-1.988461, -2.559806, 4.607293, 26.157838], ### 5022.512 
[-2.007131, -2.503352, 6.750622, 17.139043], ### 3271.366 
])


cutoff_year = 2018

# for cutoff_resolution in [2.,2.25,2.5,2.75,3.,3.25,3.5,3.75,4.]:
# for cutoff_resolution in [3.2,]:
cutoff_resolution = 3.2

P_dna = [[] for _ in range(len(res_dna))]
P_rna = [[] for _ in range(len(res_rna))]
P_aa  = [[] for _ in range(len(res_aa))]
P_ns  = [[] for _ in range(len(res_ns))]
from tqdm import tqdm
for i in tqdm(range(len(deposit_date))):
	y,m,d = deposit_date[i].split('-')
	if int(y) >= cutoff_year:
		if resolution[i] <= cutoff_resolution:
			pp = P_res[i].copy()
			pp[~np.isfinite(pp)] = 0.#np.random.rand(np.sum(~np.isfinite(pp)))*1e-6
			pp[np.isnan(pp)] = 0.#np.random.rand(np.sum(np.isnan(pp)))*1e-6

			## extract groups
			for j in range(len(res_dna)):
				keep = residues[i] == res_dna[j]
				if np.sum(keep)>0:
					P_dna[j].append(pp[keep])
			for j in range(len(res_rna)):
				keep = residues[i] == res_rna[j]
				if np.sum(keep)>0:
					P_rna[j].append(pp[keep])
			for j in range(len(res_aa)):
				keep = residues[i] == res_aa[j]
				if np.sum(keep)>0:
					P_aa[j].append(pp[keep])
			for j in range(len(res_ns)):
				keep = residues[i] == res_ns[j]
				if np.sum(keep)>0:
					P_ns[j].append(pp[keep])


####
# print('****** %f ******'%(cutoff_resolution))


# ### TEST LIMITS
# for i in range(len(P_aa)):
# 	if len(P_aa[i]) > 500:
# 		P_aa[i] = P_aa[i][:500]


keep = np.array([len(pi) > 2 for pi in P_aa])
P_aa = [P_aa[i] for i in range(len(keep)) if keep[i]]
ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_aa, 'figures/models/hmodel_residue_aa.hdf5',nres=5,maxiter=1000)

x_rg= np.array((
1.4997, #ALA
2.9268, #ARG
2.0322, #ASN
2.0219, #ASP
1.8855, #CYS
2.3649, #GLN
2.3329, #GLU
1.3694, #GLY
2.3351, #HIS
2.0084, #ILE
2.1225, #LEU
2.6013, #LYS
2.3781, #MET
2.4702, #PHE
1.7085, #PRO
1.6629, #SER
1.7735, #THR
2.7436, #TRP
2.7279, #TYR
1.7858, #VAL
))


x_mw = np.array((71.0779,156.18568,114.10264,115.0874,103.1429,128.12922,129.11398,57.05132,137.13928,113.15764,113.15764,128.17228,131.19606,147.17386,97.11518,87.0773,101.10388,186.2099,163.17326,99.13106,))
# order = np.array([ 7, 0, 15, 14, 19, 16, 4, 10, 9, 2, 3, 5, 11, 6, 12, 8, 13, 1, 18, 17])

x_size = x_rg
order = x_size.argsort()

# for i in range(order.size):
	# print(i,order[i],res_aa[order[i]])
	# print(res_aa[i])


fig,ax = cf.make_fig_set_abtautoo(x_size[order],mu_as[order],mu_bs[order],tau_as[order],tau_bs[order],covs[order])
x_ticks = np.linspace(10**(np.log10(x_size.min())//1),10**(1.+np.log10(x_size.max())//1),4)
x_ticks = np.linspace(1.,3.,5)
x_min = x_size.min()*.9
x_max = x_size.max()*1.1
for aa in fig.axes:
	# aa.set_xlim(x_size.min(),x_size.max())
	aa.set_xticks(x_ticks)
	aa.set_xticklabels(x_ticks)
	aa.set_xlim(x_min,x_max)
	# aa.set_xticklabels([res_aa[i] for i in order],rotation=90)
	

x_data = np.array([x_size[orderi] for orderi in order])
y_data = np.array([1./(1.+np.exp(mu_bs[orderi])/np.exp(mu_as[orderi])) for orderi in order])
labels = np.array([res_aa[orderi] for orderi in order])
x_label,y_label,record = langevin_labels(labels,x_data,y_data,kbT=.1,l0=.15,k0=10,epsilon=1e-13,dt=.1,gamma = 10,nsteps=4000)
# xlabel = record[-100:,:,0].mean(0)
# ylabel = record[-100:,:,1].mean(0)
# for i in range(record.shape[1]):
	# ax['P'].plot(record[:,i,0],record[:,i,1])

# ax.plot(x_data,y_data,'-o',color='k',lw=2)
for i in range(len(labels)):
	ax['P'].annotate(
		labels[i],
		xy = (x_data[i],y_data[i]),
		xytext = (x_label[i],y_label[i]),
		xycoords='data',
		textcoords='data',
		arrowprops=dict(arrowstyle="->", connectionstyle="arc3")
	)
	
ax['T'].set_ylim(ax['T'].get_xlim()[0],40.)

ax['P'].set_ylabel(r'$\langle P \rangle_{res}$')
ax['P'].set_xlabel(r'$R_g (\AA)$')
ax['B'].set_xlabel(r'$R_g (\AA)$')
ax['T'].set_xlabel(r'$R_g (\AA)$')
# ax['P'].grid(axis='x')
ax['A'].yaxis.set_major_formatter(plt.ScalarFormatter())
ax['B'].yaxis.set_major_formatter(plt.ScalarFormatter())
ax['A'].yaxis.set_minor_formatter(plt.ScalarFormatter())
ax['B'].yaxis.set_minor_formatter(plt.ScalarFormatter())
fig.savefig('figures/rendered/EPres_residue_aa_ordered.png',dpi=300)
fig.savefig('figures/rendered/EPres_residue_aa_ordered.pdf')
plt.close()


