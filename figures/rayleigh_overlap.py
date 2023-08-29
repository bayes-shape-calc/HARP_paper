import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.colors import ListedColormap
from matplotlib.colors import LightSource

def gaussian2d(gx,gy,m1,m2,s):
	return 1/(2*np.pi*s*s)*np.exp(-.5/s/s*((gx-0)**2.+(gy-m1)**2.)) + 1/(2*np.pi*s*s)*np.exp(-.5/s/s*((gx-0)**2.+(gy-m2)**2.))

def degrid(a):
	a.grid(False)

	a.xaxis.pane.fill = False
	a.yaxis.pane.fill = False
	a.zaxis.pane.fill = False
	a.xaxis.pane.set_visible(False)
	a.yaxis.pane.set_visible(False)
	a.zaxis.pane.set_visible(False)

	a.xaxis.line.set_visible(False)
	a.yaxis.line.set_visible(False)
	a.zaxis.line.set_visible(False)
	a.set_xticks((),())
	a.set_yticks((),())
	a.set_zticks((),())
	return a

N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(100/256, 133/256, N)
vals[:, 1] = np.linspace(149/256, 175/256, N)
vals[:, 2] = np.linspace(237/256, 255/256, N)
newcmp = ListedColormap(vals)

azim = 180+30
elev = 30.
xs = np.linspace(-3.5,3.5,1000)
ys = np.linspace(-4,4,1000)
gx,gy = np.meshgrid(xs,ys)

ls = LightSource(240, 30)

m1 = -1.54/2.
m2 = 1.54/2.


for sigma in [.25,.5,.75,1.0]:
	z = gaussian2d(gx,gy,m1,m2,sigma)
	rgb = ls.shade(z, cmap=newcmp, vert_exag=1., blend_mode='soft')
	
	fig = plt.figure(figsize=(4,3),dpi=600)
	ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
	ax.view_init(elev=elev, azim=azim)

	ax.set_xlim(-3.3, 3.3)
	ax.set_ylim(-3.3, 3.3)
	ax.set_zlim(0, .5)

	ax.plot_surface(gx,gy,z, facecolors=rgb,rcount=100,ccount=100,linewidth=0.1, antialiased=True, shade=False,alpha=1.)
	ax = degrid(ax)
	plt.tick_params(axis='both', which='major', direction='out', labelsize=7)
	fig.savefig('figures/rendered/ADF_%d.png'%(int(sigma*100)),transparent=True)
	plt.close()

