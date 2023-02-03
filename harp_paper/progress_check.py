## 23
import os
from .common_paper import logger,open_list,check_results,save_list,touch
import harp


def db_status(fdir):
	'''
	Will look into how complete the DB is for all of the PDB_ID list files in the DB
	'''

	fs = []
	for f in os.listdir(fdir):
		if f.startswith('PDB_ID') and f.endswith('.dat'):
			fs.append([os.path.join(fdir,f),float(f.split('_')[-1].split('A.dat')[0])])


	issues = []
	for fn,res in fs:
		pdblist = open_list(fn)

		good_cif = 0
		good_map = 0
		cif_size = 0.
		map_size = 0.
		total = len(pdblist)

		for pdbid in pdblist:
			local_cif = harp.io.rcsb.path_cif_local(pdbid,fdir)
			emdbid = harp.io.mmcif.find_emdb(local_cif)
			local_map = harp.io.rcsb.path_map_local(emdbid,fdir)
			flagcif = False
			flagmap = False
			if os.path.isfile(local_cif):
				good_cif += 1
				cif_size += os.path.getsize(local_cif)/1000000.
				flagcif = True
			if os.path.isfile(local_map):
				good_map += 1
				map_size += os.path.getsize(local_map)/1000000.
				flagmap = True
			if not (flagcif and flagmap):
				issues.append([flagcif,flagmap,pdbid,emdbid])
		print('=====================')
		print(fn)
		print('mmcif: %d/%d. %.1f Mb'%(good_cif,total,cif_size))
		print('  map: %d/%d. %.1f Mb'%(good_map,total,map_size))
		print('Issues:',issues)
		print('\n')



def progress_check(fdir,odir,cutoff):
	'''
	Takes: a float cutoff value
	Returns: a filename where the list of PDB IDs is written
	'''

	log = logger('log_4_check_progress.log')
	fname = os.path.join(fdir,'PDB_ID_EM_%.2fA.dat'%(cutoff))

	### Download the mmcifs
	pdbids = open_list(fname)

	cifs = []
	maps = []

	total = len(pdbids)
	for i in range(total):
		pdbid = pdbids[i]
		local_cif = harp.io.rcsb.path_cif_local(pdbid,fdir)
		tcif = os.path.exists(local_cif)
		if not tcif:
			local_cif2 = os.path.join(fdir,'%s.cif.gz'%(pdbid))
			tcif = os.path.exists(local_cif2)
			if tcif:
				os.system('mv %s %s'%(local_cif2,local_cif))
		tcif = touch(local_cif)
		tden = False
		try:
			emdbid = harp.io.mmcif.find_emdb(local_cif)
			local_density = harp.io.rcsb.path_map_local(emdbid,fdir)
			# tden = os.path.exists(local_density)
			# if not tden:
				# maps.append(emdbid)
			tden = touch(local_density)
			if not tden:
				maps.append(emdbid)
		except:
			pass

		if not tcif: cifs.append(pdbid)

		check = check_results(os.path.join(odir,'result_%s.csv'%(pdbid)))
		pcheck = os.path.exists(os.path.join(odir,'dict_%s.pickle'%(pdbid)))

		if not (tcif and tden and check and pcheck):
			log('%s: %s %s %s %s ----- %d/%d'%(pdbid,str(tcif),str(tden),str(check),str(pcheck),i+1,total))

	# with open('wget_cifs.sh','w') as f:
	# 	for pdbid in cifs:
	# 		f.write('wget https://files.rcsb.org/download/%s.cif.gz\n'%(pdbid))
	# 		f.write('mv %s.cif.gz %s.cif.gz\n'%(pdbid,pdbid.lower()))
	# with open('wget_maps.sh','w') as f:
	# 	for emdbid in maps:
	# 		f.write('wget ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-%s/map/emd_%s.map.gz\n'%(emdbid,emdbid))
	# os.system('chmod 744 wget_cifs.sh')
	# os.system('chmod 744 wget_maps.sh')
	log('==== Done ====')
