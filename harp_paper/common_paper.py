## 23
import os
import gzip
import numpy as np
import time
import h5py as h


def dir_path(string):
	if os.path.isdir(string):
		return string
	else:
		err = "\n\nError: output directory \"%s\" is not a directory that exists yet. Make it?\n"%(string)
		raise NotADirectoryError(err)

def clean_empty(fdir):
	fns = os.listdir(fdir)
	removed = []
	for fn in fns:
		fdn = os.path.join(fdir,fn)
		if os.path.getsize(fdn) == 0:
			os.remove(fdn)
			removed.append(fn)
	print(removed)
	print('removed %d files'%(len(removed)))

def touch(fname):
	success = False
	try:
		if os.path.exists(fname):
			g = gzip.open(fname,'r')
			g.close()
			success = True
	except:
		pass
	return success

def open_list(fname):
	out = []
	with open(fname,'r') as f:
		t = f.readline()
		for line in f:
			if line[0]=='#':
				continue
			else:
				out.append(line.replace('\n','').upper())
	return out

def save_list(fname,pdbids):

	with open(fname,'w') as f:
		f.write("# %s\n"%(str(time.ctime())))
		f.write('\n'.join(pdbids))

def check_results(path):
	success = False
	if os.path.exists(path):
		with open(path,'r') as f:
			for line in f:
				pass
			if line == '==== Done ====':
				success = True
	return success

def silent(*args,**kwargs):
	return

def clean_files(local_cif,local_density):
	if not touch(local_cif):
		try:
			os.remove(local_cif)
		except:
			pass
	if not touch(local_density):
		try:
			os.remove(local_density)
		except:
			pass

def write_results_hdf5(results,filename='./all_results.hdf5'):
	'''
	results is a dictionary
	'''

	pdbids = results.keys()
	with h.File(filename, 'w') as f:
		for pdbid in pdbids:
			complete,chains,resids,authids,resnames,probs,d = results[pdbid]
			chains = chains.astype(np.string_)
			resids = resids.astype(np.int)
			authids = authids.astype(np.string_)
			resnames = resnames.astype(np.string_)
			g = f.create_group(pdbid)
			g.attrs['_complete'] = complete
			for key in d.keys():
				g.attrs[key] = d[key]
			dset = g.create_dataset("chains",chains.shape,dtype=chains.dtype,data=chains,compression='gzip',compression_opts=9)
			dset = g.create_dataset("resids",resids.shape,dtype=resids.dtype,data=resids,compression='gzip',compression_opts=9)
			dset = g.create_dataset("authids",authids.shape,dtype=authids.dtype,data=authids,compression='gzip',compression_opts=9)
			dset = g.create_dataset("resnames",resnames.shape,dtype=resnames.dtype,data=resnames,compression='gzip',compression_opts=9)
			dset = g.create_dataset("probs",probs.shape,dtype=probs.dtype,data=probs,compression='gzip',compression_opts=9)


def write_csv_results(path_out,mol,probs,keep):
	chains = mol.chain[keep]
	resids = mol.resid[keep]
	authids = mol.authresid[keep]
	resnames = mol.resname[keep]
	ps = probs[keep]
	with open(path_out,'w') as f:
		f.write('Chain,Residue ID,Auth ID,Residue Name,P_atomic\n')
		for i in range(chains.size):
			f.write('%s,%s,%s,%s,%.5f\n'%(chains[i],resids[i],authids[i],resnames[i],ps[i]))
		f.write('==== Done ====')

def read_csv_results(path_in):
	chains = []
	resids = []
	authids = []
	resnames = []
	probs = []
	complete = False
	with open(path_in,'r') as f:
		f.readline() ## skip header
		for line in f:
			if line[-1] == '\n':
				line = line[:-1]
			if line[0] == '=':
				if '==== Done ====':
					complete = True
			else:
				chain,resid,authid,resname,prob = line.split(',')
				chains.append(chain)
				authids.append(authid)
				resids.append(resid)
				resnames.append(resname)
				probs.append(float(prob))
	if complete:
		return complete,np.array(chains),np.array(resids),np.array(authids),np.array(resnames),np.array(probs)
	return complete,None,None,None,None



class logger(object):
	'''
	Create logger object, then make calls to print to internal string
	Output log with .print, or .write(filename)
	'''
	def __init__(self,fname=None):
		self.s = ''
		self.fname = fname
		self.flag_file = False if self.fname is None else True
		if self.flag_file: ## initialize everything
			self.write(self.fname)

	def __call__(self,*args):
		line = ' '.join([str(arg) for arg in args]) + '\n'
		self.s += line
		if self.flag_file:
			with open(self.fname,'a') as f:
				f.write(line)

	def write(self,fname):
		with open(fname,'w') as f:
			f.write(self.s)

	def print(self):
		print(self.s)
