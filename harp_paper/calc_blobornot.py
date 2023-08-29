## 23
import os
import time
import numpy as np
import multiprocessing
import harp
from .common_paper import open_list,logger,clean_files,write_csv_results,check_results

def run_em_pdbid(pdbid,fdir,job_i,total,out_dir,allow_big):
	t0 = time.time()

	## Setup log
	log = logger(os.path.join(out_dir,'log_%s.log'%(pdbid)))
	log('------------ %s: %d/%d'%(pdbid,job_i,total))

	## Load data into memory
	local_cif = harp.io.rcsb.path_cif_local(pdbid,fdir)
	emdbid = harp.io.mmcif.find_emdb(local_cif)
	local_density = harp.io.rcsb.path_map_local(emdbid,fdir)

	try:
		filesize = os.stat(local_density).st_size/(1e6)
		if filesize > 2000. and (not allow_big):
			log('EM map file is too large: %f Mb'%(filesize))
			log('==== Skip  ====')
			return
		
		mol = harp.molecule.load(local_cif,only_polymers=True)
		grid,density = harp.density.load(local_density)
		log('Loaded %s'%(pdbid))
		log('Grid: %s %s'%(str(grid.nxyz),str(grid.dxyz)))
		log('Density memory: %.2f Mb'%(density.nbytes/(1e6)))
	except Exception as e: ## Maybe it was a partial file so remove both
		log('Load data failure')
		log(str(e))
		clean_files(local_cif,local_density)
		return

	## Perform calculation
	harp.models.use_c()
	try:
		probs, ln_evidence = harp.bayes_model_select.bms_molecule(grid, density, mol, emit=log)
		keep = np.zeros(probs.size,dtype='bool')
		for chain in mol.unique_chains:
			subchain = mol.get_chain(chain)
			for i in range(subchain.unique_residues.size):
				resi = subchain.unique_residues[i]
				subresidue = subchain.get_residue(resi)
				keep[mol.atomid==subresidue.atomid[0]] = True
		log("<P_not>: %.4f"%(np.nanmean(probs[keep])))
		log("Finite: %d / %d"%(np.isfinite(probs[keep]).sum(),keep.sum()))

		## Write info to a CSV file
		path_out = os.path.join(out_dir,'result_%s.csv'%(pdbid))
		log('Result path: %s'%(path_out))
		write_csv_results(path_out,mol,probs,keep)

		## Write log information to file
		t1 = time.time()
		log('Total time: %.2f sec'%(t1-t0))
		log('==== Done ====')
	except Exception as e:
		log('==== Fail ====')
		log(str(e))
	return

def calc_blobornot(fdir,out_dir,cutoff,num_workers):
	t0 = time.time()
	log = logger('log_2_calculation.log')

	num_workers = np.min((num_workers,multiprocessing.cpu_count()))
	report_delay = 10.0 ## sec
	loop_delay = 1.0 ## sec

	fname = 'PDB_ID_EM_%.2fA.dat'%(cutoff)
	fname = os.path.join(fdir,fname)

	if not os.path.exists(out_dir):
		os.mkdir(out_dir)





	## first pass is 'small' with multiprocessing, second pass is 'large' with one cpu to avoid memory issues 
	for nw,ab,logit,msg in zip([num_workers,1],[False,True],[True,False],['Small','Large']):
		pdbids = open_list(fname)
		pdbids = pdbids[::-1]
		log("%s\nProcessing %d PDBs (%s)\n==================================="%(str(time.ctime()),len(pdbids),msg))
		job_i = 0
		job_total = len(pdbids)
		print('PASS',nw,ab,logit)
		jobs = []
		t1 = time.time()
		while True:
			if len(pdbids) == 0 and len(jobs) == 0:
				log('===================================')
				log('Completed all jobs')
				break
			while len(jobs) < nw:
				if len(pdbids) > 0:
					pdbid = pdbids.pop()
					if not check_results(os.path.join(out_dir,'result_%s.csv'%(pdbid))):
						job_i += 1
						if logit: log("Starting %s at %s - %d/%d"%(pdbid,str(time.ctime()),job_i,job_total))
						p = multiprocessing.Process(target=run_em_pdbid, args=(pdbid,fdir,job_i,job_total,out_dir,ab))
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

	t3 = time.time()
	log(str(time.ctime()))
	log('<t> = %.3f s'%((t3-t0)/float(job_total)))
	log('==== Done ====')
