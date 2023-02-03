'''
Downloads the cryoEM files from the PDB
'''
import os
import json
import gzip
import time
import requests
from .common_paper import logger,open_list,check_results,save_list,touch
import harp

def query_rcsb(exptl_method='ELECTRON MICROSCOPY',resolution_max=8.0,cutdate='2023-01-01'):
	'''
	Get file lists from the RCSB
	Follow search API guide at https://search.rcsb.org/index.html
	'''
	query = {
		"query":{"type":"group","logical_operator":"and","nodes":[
			{"type":"terminal","service":"text","parameters":{"attribute":"exptl.method","operator":"exact_match","negation":False,"value":"ELECTRON MICROSCOPY"}},
			{"type":"terminal","service":"text","parameters":{"attribute":"em_3d_reconstruction.resolution","operator":"less_or_equal","negation":False,"value":resolution_max}},
			{"type":"terminal","service":"text","parameters":{"attribute":"rcsb_accession_info.has_released_experimental_data","operator":"exact_match","negation":False,"value":"Y"}},
			{"type":"terminal","service":"text","parameters":{"attribute":"rcsb_accession_info.initial_release_date","operator":"greater","negation":False,"value":"1900-01-01"}},
			{"type":"terminal","service":"text","parameters":{"attribute":"rcsb_accession_info.initial_release_date","operator":"less","negation":False,"value":cutdate}}
		],"label":"text"},

		"return_type":"entry",
		"request_options": {"return_all_hits": True}
	}
	data_json = json.dumps(query)
	r = requests.get('https://search.rcsb.org/rcsbsearch/v2/query?json=%s'%data_json)
	if r.status_code == 200:
		d = json.loads(r.text)
		pdbids = [dd['identifier'] for dd in d['result_set']]
		return pdbids,d
	raise Exception("failure, %d"%(r.status_code))

def download_rcsb(fdir,cutoff,cutdate='2023-01-01'):
	'''
	Takes: a float cutoff value
	Returns: a filename where the list of PDB IDs is written
	'''

	log = logger('log_1_download_rcsb.log')

	if not os.path.isdir(fdir):
		os.mkdir(fdir)

	fname = 'PDB_ID_EM_%.2fA.dat'%(cutoff)
	fname = os.path.join(fdir,fname)

	### Query the pdb for the entry list
	log(str(time.ctime()))
	if not os.path.exists(fname):
		pdbids_em,d = query_rcsb(exptl_method='ELECTRON MICROSCOPY',resolution_max=cutoff,cutdate='2023-01-01')
		# pdbids_xray_8,d = query_rcsb(exptl_method='X-RAY DIFFRACTION',resolution_max=8.)
		log('Downloaded list of %d PDB IDs from RCSB'%(len(pdbids_em)))

		#### Blacklist problem structures
		## 3IZO - the map has been pulled from the EMDB
		blacklist = ['3IZO',]
		for bli in blacklist:
			if pdbids_em.count(bli)>0:
				pdbids_em.remove(bli)

		save_list(fname,pdbids_em)
		log('Saved list in %s'%(fname))


	### Download files for the entry list into fdir
	pdbids = open_list(fname)

	total = len(pdbids)
	for i in range(total):
		t0 = time.time()
		pdbid = pdbids[i]
		log('%s--------------------------------------------------------- %d/%d'%(pdbid,i,total))

		### Get the mmcif
		local_cif = harp.io.rcsb.path_cif_local(pdbid,fdir)
		if touch(local_cif):
			log('mmcif exists at %s'%(local_cif))
		else:
			ftp_cif = harp.io.rcsb.path_cif_ftp(pdbid,fdir)
			harp.io.rcsb.download_wrapper(ftp_cif,local_cif,True,True,log)
			log('downloaded mmcif from %s'%(ftp_cif))
		if not touch(local_cif):
			continue

		### Get the mrc map
		emdbid = harp.io.mmcif.find_emdb(local_cif)
		local_density = harp.io.rcsb.path_map_local(emdbid,fdir)
		if touch(local_density):
			log('%s map exists at %s'%(emdbid,local_density))
		else:
			ftp_density = harp.io.rcsb.path_map_ftp(emdbid,fdir)
			harp.io.rcsb.download_wrapper(ftp_density,local_density,True,True,log)
			log('downloaded %s map from %s'%(emdbid,ftp_density))

		t1 = time.time()
		cif_size = os.stat(local_cif).st_size/(1e6) if os.path.exists(local_cif) else 0.
		density_size = os.stat(local_density).st_size/(1e6) if os.path.exists(local_density) else 0.
		log('Downloaded %s in %.2f sec. (mmCIF: %.2f Mb; Map: %.2f Mb).'%(pdbid,(t1-t0),cif_size,density_size))

	log('==== Done ====')
