## 23

import os
import time
import numpy as np
import multiprocessing
import harp
from .common_paper import open_list,logger,clean_files,write_csv_results,check_results,touch


## calculate distance of each residue from the COM of the entire molecule
def run_radial_pdbid(pdbid,fdir,job_i,total,out_dir,allow_big):
	t0 = time.time()

	## Load data into memory
	local_cif = harp.io.rcsb.path_cif_local(pdbid,fdir)
	emdbid = harp.io.mmcif.find_emdb(local_cif)
	local_density = harp.io.rcsb.path_map_local(emdbid,fdir)

	try:
		mol = harp.molecule.load(local_cif)
		mol = mol.remove_hetatoms()
	except Exception as e: ## Maybe it was a partial file so remove both
		clean_files(local_cif,local_density)
		return

	## Perform calculation
	try:
		keep = np.bitwise_or(mol.atomname=='\"C4\'\"',mol.atomname=='CA')
		last_resid = -1
		last_chain = ''
		for i in range(keep.size):
			if keep[i]:
				if last_resid == mol.resid[i] and last_chain == mol.chain[i]:
					keep[i] = False
				else:
					last_resid = mol.resid[i]
					last_chain = mol.chain[i]
		kept = mol.get_set(keep)
		mag_r = np.linalg.norm(kept.xyz-mol.com()[None,:],axis=1)

		## Write info to a CSV file
		path_out = os.path.join(out_dir,'radial_%s.npy'%(pdbid))
		np.save(path_out,mag_r)

		## Write log information to file
		t1 = time.time()
	except Exception as e:
		print(str(e))
	return

def calc_radial(fdir,out_dir,cutoff,num_workers,allow_big=False):
	t0 = time.time()
	log = logger('log_2B_radial.log')

	num_workers = np.min((num_workers,multiprocessing.cpu_count()))
	report_delay = 10.0 ## sec
	loop_delay = 1.0 ## sec

	fname = 'PDB_ID_EM_%.2fA.dat'%(cutoff)
	fname = os.path.join(fdir,fname)

	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	pdbids = open_list(fname)
	pdbids = pdbids[::-1]

	log("%s\nProcessing %d PDBs\n==================================="%(str(time.ctime()),len(pdbids)))

	job_i = 0
	job_total = len(pdbids)
	jobs = []
	t1 = time.time()
	while True:
		if len(pdbids) == 0 and len(jobs) == 0:
			log('===================================')
			log('Completed all jobs')
			break
		while len(jobs) < num_workers:
			if len(pdbids) > 0:
				pdbid = pdbids.pop()
				if not touch(os.path.join(out_dir,'radial_%s.npy'%(pdbid))):
					job_i += 1
					log("Starting %s at %s - %d/%d"%(pdbid,str(time.ctime()),job_i,job_total))
					p = multiprocessing.Process(target=run_radial_pdbid, args=(pdbid,fdir,job_i,job_total,out_dir,allow_big))
					jobs.append([p,pdbid,time.time()])
					p.start()
				else:
					job_i += 1
			else:
				break

		t2 = time.time()
		if t2-t1 > report_delay:
			log('--> '+',\t'.join(["%s %.2fs"%(str(job[1]),t2-job[2]) for job in jobs]))
			t1 = t2
		jobs = [job for job in jobs if job[0].is_alive()]
		time.sleep(loop_delay)


	import h5py
	print('Collecting all files into all_radial.hdf5')
	pdbids = open_list(fname)
	pdbids = pdbids[::-1]
	with h5py.File(os.path.join(out_dir,'all_radial.hdf5'),'w') as f:
		for pdbid in pdbids:
			fpath = os.path.join(out_dir,'radial_%s.npy'%(pdbid))
			if touch(fpath):
				g = f.create_group(pdbid)
				radial = np.load(fpath)
				dset = g.create_dataset("radial",radial.shape,dtype=radial.dtype,data=radial,compression='gzip',compression_opts=9)
			else:
				print('fail',pdbid)
	print('done')
	t3 = time.time()
	log(str(time.ctime()))
	log('<t> = %.3f s'%((t3-t0)/float(job_total)))
	log('==== Done ====')
