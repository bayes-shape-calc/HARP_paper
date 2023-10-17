import os
import time
import numpy as np
import multiprocessing
import harp
from .common_paper import open_list,logger,clean_files,write_csv_results,check_results,touch
import numba as nb

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


## calculate distance of each residue from the COM of the entire molecule
def run_disthist_pdbid(pdbid,fdir,job_i,total,out_dir):
	t0 = time.time()

	## Load data into memory
	local_cif = harp.io.rcsb.path_cif_local(pdbid,fdir)
	emdbid = harp.io.mmcif.find_emdb(local_cif)
	local_density = harp.io.rcsb.path_map_local(emdbid,fdir)

	try:
		mol = harp.molecule.load(local_cif,only_polymers=True)
		# mol = mol.remove_hetatoms()
	except Exception as e: ## Maybe it was a partial file so remove both
		clean_files(local_cif,local_density)
		return

	## Perform calculation
	try:
		coms = []
		for chain in mol.unique_chains:
			subchain = mol.get_chain(chain)
			for residue in subchain.unique_residues:
				subresidue = subchain.get_residue(residue)
				coms.append(subresidue.com())
		coms = np.array(coms)
				
		xleft,hist = hist_distances_closest(coms,0.,20.,2000)
		out = np.array((xleft,hist))
		# dx = xleft[1]-xleft[0]
		# xmid = xleft + .5*dx
		# print(np.sum(xmid*hist)/np.sum(hist))

		## Write info to a binary file
		path_out = os.path.join(out_dir,'disthist_%s.npy'%(pdbid))
		np.save(path_out,out)

		t1 = time.time()
		
	except Exception as e:
		print(str(e))
	return

def calc_disthist(fdir,out_dir,cutoff,num_workers):
	t0 = time.time()
	log = logger('log_4_disthist.log')

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
				if not touch(os.path.join(out_dir,'disthist_%s.npy'%(pdbid))):
					job_i += 1
					log("Starting %s at %s - %d/%d"%(pdbid,str(time.ctime()),job_i,job_total))
					p = multiprocessing.Process(target=run_disthist_pdbid, args=(pdbid,fdir,job_i,job_total,out_dir))
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
	print('Collecting all files into all_disthist.npy')
	pdbids = open_list(fname)
	pdbids = pdbids[::-1]

	out = []
	for pdbid in pdbids:
		fpath = os.path.join(out_dir,'disthist_%s.npy'%(pdbid))
		if touch(fpath):
			dh = np.load(fpath)
			out.append(dh[1])
		else:
			print('fail',pdbid)
	out = [dh[0],]+out
	out = np.array(out)
	np.save(os.path.join(out_dir,'all_disthist.npy'),out)
	import shutil
	shutil.copy(os.path.join(out_dir,'all_disthist.npy'),'./all_disthist.npy')
	print('done')
	t3 = time.time()
	log(str(time.ctime()))
	log('<t> = %.3f s'%((t3-t0)/float(job_total)))
	log('==== Done ====')
