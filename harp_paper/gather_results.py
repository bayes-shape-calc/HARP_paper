import numpy as np
import os
import pickle
from .common_paper import open_list,logger,read_csv_results,write_results_hdf5

def get_fnames(pdbid):
	return os.path.join(odir,'dict_%s.pickle'%(pdbid)),os.path.join(odir,'result_%s.csv'%(pdbid))

def gather_results(fdir,odir,cutoff):
	log = logger('log_5_collect_results.log')

	fname = os.path.join(fdir,'PDB_ID_EM_%.2fA.dat'%(cutoff))
	pdbids = open_list(fname)

	out = {}
	total = len(pdbids)
	for i in range(total):
		pdbid = pdbids[i]
		try:
			with open(os.path.join(odir,'dict_%s.pickle'%(pdbid)),'rb') as f:
				d = pickle.load(f)
			complete,chains,resids,authids,resnames,probs = read_csv_results(os.path.join(odir,'result_%s.csv'%(pdbid)))

			# try:
			# 	with open(os.path.join(odir,'alignment_%s.val'%(pdbid)),'r') as f:
			# 		d['alignment'] = float(f.read())
			# except:
			# 	d['alignment'] = -1.

			out[pdbid] = [complete,chains,resids,authids,resnames,probs,d]
			log('%04d/%04d - %s %s'%(i,total-1,pdbid,str(complete)))
		except:
			log('%04d/%04d - %s %s'%(i,total-1,pdbid,'FAIL'))



	write_results_hdf5(out,filename=os.path.join(odir,'./all_results.hdf5'))
