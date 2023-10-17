import os
import time
import numpy as np
import pickle
from .common_paper import save_list,open_list,touch,logger
import harp

entries ={
	'microscope':'_em_imaging.microscope_model',
	'mag':'_em_imaging.calibrated_magnification',
	'voltage':'_em_imaging.accelerating_voltage',
	'cs':'_em_imaging.nominal_cs',

	'camera':'_em_image_recording.film_or_detector_model',
	'dose':'_em_image_recording.avg_electron_dose_per_image',
	'exposure':'_em_image_recording.average_exposure_time',
	'num imgs':'_em_image_recording.num_real_images',
	'num frames':'_em_image_scans.frames_per_image',

	'resolution':'_em_3d_reconstruction.resolution',
	'resolution method':'_em_3d_reconstruction.resolution_method',
	'num particles':'_em_3d_reconstruction.num_particles',
	'num particles selected':'_em_particle_selection.num_particles_selected',

	'citation year':'_citation.year',
	'citation journal':'_citation.journal_abbrev',

	'PDB ID':'_entry.id',
	'method':'_exptl.method',
	'EMDB ID':'_pdbx_database_related.db_id',
	'deposit date':'_pdbx_database_status.recvd_initial_deposition_date',
	'status':'_pdbx_database_status.status_code',

	'organism':'_em_entity_assembly_naturalsource.organism',
	'organism id':'_em_entity_assembly_naturalsource.ncbi_tax_id',
	# 'organism produced in':'_entity_src_gen.pdbx_host_org_scientific_name',
	# 'organism produced in ID':'_entity_src_gen.pdbx_host_org_ncbi_taxonomy_id',
	'pdb keywords':'_struct_keywords.pdbx_keywords',

	'pH':'_em_buffer.pH',
	'cryogen':'_em_vitrification.cryogen_name',
	'humidity':'_em_vitrification.humidity',

	'formula weight':'_entity.formula_weight',
	'num fw':'_entity.pdbx_number_of_molecules',
	'type fw':'_entity.type',
	'molwt':'_em_entity_assembly_molwt.value',
	'molwt units':'_em_entity_assembly_molwt.units',

	'protein':'_entity_poly.type',
	'rna':'_entity_poly.type',
	'dna':'_entity_poly.type',

	'software category':'_em_software.category',
	'software name':'_em_software.name',
	'software version':'_em_software.version',
}

def clean_string(s):
	if type(s) is str:
		s = s.replace('\'','')
		s = s.replace('\"','')
	if type(s) is list:
		for i in range(len(s)):
			if type(s[i]) is str:
				s[i] = s[i].replace('\'','')
				s[i] = s[i].replace('\"','')
	return s

def process_entries(d):
	'''
	Only take the first experiments if there are multiple experiment entries
	'''

	for entry in entries.keys():
		d[entry] = clean_string(d[entry])

	try:
		from dateutil.parser import parse
		d['deposit year'] = parse(d['deposit date'], fuzzy=True).year
	except:
		d['deposit year'] = 'missing entry'

	if type(d['citation journal']) is list: ## just take the first one....?
		d['citation journal'] = d['citation journal'][0]
		d['citation year'] = d['citation year'][0]

	try:
		if type(d['formula weight']) is list:
			t = np.array([dd == 'polymer' for dd in d['type fw']]).astype('int')
			n = np.array([int(dd) for dd in d['num fw']])
			fw = np.array([float(dd) for dd in d['formula weight']])
			d['formula weight'] = np.sum(fw*n*t)
		else:
			d['formula weight'] = int(d['num fw'])*float(d['formula weight'])
		del d['type fw']
		del d['num fw']
	except:
		d['formula weight'] = 'missing entry'


	for i in ['category','name','version']:
		if not type(d['software %s'%(i)]) is list:
			d['software %s'%(i)] = list(d['software %s'%(i)])
		d['software %s'%(i)] = clean_string(d['software %s'%(i)])

	for cat in ['IMAGE ACQUISITION','PARTICLE SELECTION','CTF CORRECTION','MODEL FITTING','INITIAL EULER ASSIGNMENT','FINAL EULER ASSIGNMENT','CLASSIFICATION','RECONSTRUCTION','MODEL REFINEMENT']:
		b = [dd == cat for dd in d['software category']]
		sn = 'missing entry'
		sv = 'missing entry'
		if np.any(b):
			index = np.argmax(b)
			sn = d['software name'][index]
			sv = d['software version'][index]
		d['software name - %s'%(cat.lower())] = sn
		d['software version - %s'%(cat.lower())] = sv
	del d['software category']
	del d['software name']
	del d['software version']

	for t,tt in zip(['protein','rna','dna'],['polypeptide(L)','polyribonucleotide','polydeoxyribonucleotide']):
		try:
			if not type(d[t]) is list: d[t] = [d[t]]
			d[t] = np.any([dd==tt for dd in d[t]])
		except:
			d[t] = False

	for k in ['num imgs','num frames','num particles','num particles selected','citation year','deposit year']:
		if type(d[k]) is list: d[k] = d[k][0]
		if not d[k] in ['missing entry','?','.']:
			d[k] = int(d[k])

	for k in ['mag','voltage','cs','dose','exposure','resolution','pH','humidity','molwt']:
		if type(d[k]) is list: d[k] = d[k][0]
		if not d[k] in ['missing entry','?','.']:
			d[k] = float(d[k])

	## at this point, nothing should be a list anymore.... so take the first entry/experiment in the file.
	for k in d.keys():
		if type(d[k]) is list:
			d[k] = d[k][0]

	return d

def parse_mmcif(pdbid,fdir,odir):

	local_cif = harp.io.rcsb.path_cif_local(pdbid,fdir)
	if not touch(local_cif):
		harp.io.rcsb.download_mmcif(pdbid,fdir,overwrite=False,verbose=True,emit=print)
	else:
		pass
		# print(pdbid)
	d = harp.io.mmcif.load_mmcif_dict(local_cif)
	out = {entry:'missing entry' for entry in entries.keys()}
	for entry in entries.keys():
		key =entries[entry]
		try:
			out[entry] = d[key]
		except:
			try:
				loop,tag = key.split('.')
				h,vals = d[loop]
				index = np.argmax([hh==key for hh in h])
				out[entry] = [v[index] for v in vals]
			except:
				pass
	out = process_entries(out)

	# with open(os.path.join(odir,'info_%s.csv'%(pdbid)),'w') as f:
		# for entry in sorted(list(out.keys()),key=str.lower):
			# f.write("%s,%s\n"%(entry,out[entry]))
		# f.write('==== Done ====')
	with open(os.path.join(odir,'dict_%s.pickle'%(pdbid)),'wb') as f:
		pickle.dump(out,f)

def extract_metadata(fdir,odir,cutoff,overwrite=False):

	fname = os.path.join(fdir,'PDB_ID_EM_%.2fA.dat'%(cutoff))
	log = logger('log_5_extract_mmcif.log')

	pdbids = open_list(fname)

	total = len(pdbids)
	for i in range(total):
		pdbid = pdbids[i]
		try:
			if (not overwrite) and os.path.exists(os.path.join(odir,'dict_%s.pickle'%(pdbid))):
				log('%s skip'%(pdbid))
			else:
				t0 = time.time()
				parse_mmcif(pdbid,fdir,odir)
				t1 = time.time()
				log('%s Success. %.2f sec - %d/%d'%(pdbid,t1-t0,i,total))
		except Exception as e:
			log('%s Failed. %s - %d/%d'%(pdbid,str(e),i,total))
	log('==== Done ====')
## left to do....
	# '_em_sample_support.grid_material'
	# '_em_sample_support.grid_type'

	# ### DEBUGGING
	# pdbid = '3JCN'
	# parse_mmcif(pdbid,'./paper/temp','./paper/temp')
	#
	# with open(os.path.join('./paper/temp','dict_%s.pickle'%(pdbid)),'rb') as f:
	# 	q = pickle.load(f)
	# for k in q.keys():
	# 	print(k,q[k])
