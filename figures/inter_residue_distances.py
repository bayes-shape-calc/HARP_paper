## 23

import harp
import numpy as np
import numba as nb
import matplotlib.pyplot as plt


#### Preparation
pdbid = '8b0x'
basedir = '../testdb'

success, path_mol, path_density, flag_density = harp.io.rcsb.get_pdb(pdbid,basedir,False,False,print)
mol = harp.molecule.load(path_mol)
print('--> Loaded %s'%(pdbid))

mol = mol.remove_hetatoms()


coms = []
for chain in mol.unique_chains:
	subchain = mol.get_chain(chain)
	for residue in subchain.unique_residues:
		subresidue = subchain.get_residue(residue)

		coms.append(subresidue.com())
coms = np.array(coms)

@nb.njit
def hist_distances_closest(locs,low,high,n):
	counts = np.zeros(n)
	delta = (high-low)
	dc = delta/float(n)
	xleft = low+np.arange(n)*dc#+dc*.5
	dist = 0.0
	l1 = np.zeros(3)
	l2 = np.zeros(3)

	nmol,nd = locs.shape
	closest = np.zeros(nmol)+np.inf
	for i in range(nmol):
		for k in range(3):
			l1[k] = locs[i,k]
		for j in range(nmol):
			for k in range(3):
				l2[k] = locs[j,k] - l1[k]
			dist = np.sqrt(l2[0]*l2[0]+l2[1]*l2[1]+l2[2]*l2[2])
			if dist < closest[i] and i!=j:
				closest[i] = dist
		if closest[i] >= low and closest[i] < high:
			counts[int((closest[i]-low)//dc)] += 1.0
	return xleft,counts

@nb.njit
def hist_distances_all(locs,low,high,n):
	counts = np.zeros(n)
	delta = (high-low)
	dc = delta/float(n)
	xleft = low+np.arange(n)*dc#+dc*.5
	dist = 0.0
	l1 = np.zeros(3)
	l2 = np.zeros(3)
	
	nmol,nd = locs.shape
	for i in range(nmol):
		for k in range(3):
			l1[k] = locs[i,k]
		for j in range(i+1,nmol):
			for k in range(3):
				l2[k] = locs[j,k] - l1[k]
			dist = np.sqrt(l2[0]*l2[0]+l2[1]*l2[1]+l2[2]*l2[2])
			if dist >= low and dist < high:
				counts[int((dist-low)//dc)] += 1.0
	return xleft,counts
	
print(coms.size)
xleft,hist = hist_distances_all(coms,0.,200.,10000)

fig,ax=plt.subplots(1)
axins = ax.inset_axes([.4,.08,.35,.35])
ax.plot(xleft,hist)
ax.set_xlim(0,200.)
axins.plot(xleft,hist)
xright = 15.
xind = np.argmax(xleft > xright)
xright = xleft[xind]
axins.set_xlim(0,xright)
ymax = hist[:xind].max()
axins.set_ylim(-.02*ymax,1.02*ymax)
ax.set_ylabel('Counts')
ax.set_xlabel(r'Pairwise distance between residue COM ($\AA$)')
ax.set_title(pdbid)
axins.minorticks_on()
plt.savefig('rendered/inter_residue_distances.png')
plt.savefig('rendered/inter_residue_distances.pdf')
plt.show()


xleft,hist = hist_distances_closest(coms,0.,200.,10000)
dx = xleft[1]-xleft[0]
xmid = xleft + .5*dx
print(np.sum(xmid*hist)/np.sum(hist))
fig,ax=plt.subplots(1)
ax.plot(xleft,hist)
from scipy import ndimage as nd
dd = nd.gaussian_filter1d(hist,5.)
ax.plot(xleft,dd,color='tab:red')
ax.axvline(x=xleft[dd.argmax()],color='tab:red')
ax.set_xlim(0,8)
ax.minorticks_on()
ax.set_ylabel('Counts')
ax.set_xlabel(r'Closest pairwise distance between residue COM ($\AA$)')
plt.savefig('rendered/inter_residue_distances_closest.png')
plt.savefig('rendered/inter_residue_distances_closest.pdf')
plt.show()