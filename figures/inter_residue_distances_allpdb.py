import numpy as np
import matplotlib.pyplot as plt


#### Preparation
fn = './all_disthist.npy'
d = np.load(fn)

xleft = d[0]
hist = d[1:]
dx = xleft[1]-xleft[0]
xmid = xleft + dx/2.

average_profile  = hist.mean(0)
cumulative = hist.sum(0).cumsum()/hist.sum()
normalized = hist/hist.sum(1)[:,None]
average_normalized_profile = normalized.mean(0)

average = np.sum(xleft*average_profile)/np.sum(average_profile)
average_normalized = np.sum(xleft*average_normalized_profile)/np.sum(average_normalized_profile)
median = xleft[np.argmax(cumulative>0.5)]
max_average = xleft[np.argmax(average_profile)]
max_average_normalized = xleft[np.argmax(average_normalized_profile)]


fig,ax=plt.subplots(1)
ax.plot(xleft,average_normalized_profile/average_normalized_profile.sum()/dx*dx)
plt.axvline(average_normalized,color='tab:red')
ax.set_xlim(0,8)
ax.minorticks_on()
ax.tick_params(axis='y', which='minor',left=False)
print(average_normalized)

ax.text(average_normalized, 0.98, r'$\langle r \rangle = %.2f \AA$   '%(average_normalized), color='r', ha='right', va='top', rotation=0, fontsize=10, transform=ax.get_xaxis_transform())
props = dict( facecolor='white')
ax.text(0.775, 0.975, 'N=%d\nn=%d'%(hist.shape[0],hist.sum()), transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

ax.set_ylabel('Probability')
ax.set_xlabel(r'$r$, closest pairwise distance between residue COM ($\AA$)')
plt.tight_layout()
plt.savefig('figures/rendered/inter_residue_distances_closest_allpdb.png')
plt.savefig('figures/rendered/inter_residue_distances_closest_allpdb.pdf')
plt.close()