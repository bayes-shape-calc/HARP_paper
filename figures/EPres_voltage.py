import common_figures as cf
import numpy as np
import matplotlib.pyplot as plt


deposit_date = cf.load_depositdate('./all_results.hdf5')
resolution = cf.load_resolution('./all_results.hdf5')
P_res = cf.load_P_res('./all_results.hdf5')
pdbids = cf.load_pdbids('./all_results.hdf5')

voltage = cf.load_voltage('./all_results.hdf5')
print('loaded')

for i in range(len(P_res)):
	P_res[i][np.isnan(P_res[i])] = 0.

### apply year cutoff
ycut = 2018
rescutoff = 8.
keep =[False for _ in range(len(deposit_date))]
for i in range(len(deposit_date)):
	y,m,d = deposit_date[i].split('-')
	if (int(y) >= ycut) and resolution[i] <= rescutoff:
		keep[i] = True

keep = np.bitwise_and(keep,np.bitwise_not(np.isnan(voltage)))

deposit_date = [deposit_date[i] for i in range(len(deposit_date)) if keep[i]]
P_res = [P_res[i] for i in range(len(P_res)) if keep[i]]
pdbids = [pdbids[i] for i in range(len(pdbids)) if keep[i]]
pdbids = np.array(pdbids)
voltage = voltage[keep]


voltages = np.unique(voltage)
counts = np.array([np.sum(voltage==v) for v in voltages])
keep = counts > 5
voltages = voltages[keep]
P_ress = []
for i in range(voltages.size):
	P = []
	for j in range(voltage.size):
		if voltage[j] == voltages[i]:
			P.append(P_res[j])
	P_ress.append(P)

voltages = ['%d'%(v) for v in voltages]

###  cap groups at 3000
for i in range(len(P_ress)):
	ind = 1
	while len(P_ress[i]) > 5500:
		if ind == 1:
			voltages[i] = voltages[i]+'\n(group %d)'%(ind)
		ind += 1
		rvss = np.unique(np.random.randint(low=0,high=len(P_ress[i]),size=3000))
		newp = [P_ress[i][j] for j in range(len(P_ress[i])) if j in rvss]
		P_ress.append(newp)
		voltages = np.append(voltages,'%s%d)'%(voltages[i][:-2],ind))
		P_ress[i] = [P_ress[i][j] for j in range(len(P_ress[i])) if j not in rvss]



# # ps,fig,ax = cf.process_sets_indiv(x,P_res_months)
ps,mu_as,mu_bs,tau_as,tau_bs,covs = cf.process_sets_hierarchical(P_ress,'figures/models/hmodel_voltage.hdf5',nres=5)


rx = np.arange(voltages.size)
fig,ax = cf.make_fig_set_abtautoo(rx,mu_as,mu_bs,tau_as,tau_bs,covs)

# rmax = rr[-1]
# rmin = rr[0]
# rx = np.linspace(rmin,rmax,6).astype('int')

# # rx = np.array([0,20,40,60,80,100,120])
for aa in fig.axes:
	aa.set_xticks(rx)
	aa.set_xticklabels(voltages,rotation=0)
# 	aa.set_xlim(rr[0],rr[-1])
# 	# aa.set_xlim(5,65)
# 	# aa.xaxis.major.formatter._useMathText = True
# 	# aa.ticklabel_format(axis='x',style='sci')
# 	# aa.set_xscale('log')


ax['P'].set_ylim(.4,.5)
ax['P'].set_xlabel(r'Voltage (kV)')
ax['B'].set_xlabel(r'Voltage (kV)')
ax['T'].set_xlabel(r'Voltage (kV)')


# fig.tight_layout()
fig.subplots_adjust(bottom=.22)
#
fig.savefig('figures/rendered/EPres_voltage.pdf')
fig.savefig('figures/rendered/EPres_voltage.png',dpi=300)
plt.close()
