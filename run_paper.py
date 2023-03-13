import argparse
import harp_paper
import os


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Run the HARP paper calculations")
	parser.add_argument('step', type=str, choices=['download','calculate','radial','metadata','progress','check','results','everything','remove_empty','disthists'], help='Step of the paper to run. Using everything runs: <download, calculate, metadata, alignment, progress, collect>')
	parser.add_argument('--data_dir', type=harp_paper.common_paper.dir_path, default='./', help='The directory where maps and models are stored.')
	parser.add_argument('--results_dir', type=harp_paper.common_paper.dir_path, default='./', help='The directory where caclulated results are stored.')
	parser.add_argument('--cutoff', type=float, default=8.0, help='Maximum reported resolution to include in calculations.')
	parser.add_argument('--nworkers', type=int, default=4, help='Number of CPUs to use during HARP calculations.')
	parser.add_argument('--cutoff_date',type=str, default='2023-01-01',help='Download cutoff day. Only download strucrtures released to the PDB before this date.')
	cutdate='2023-01-01'

	# parser.add_argument('--allow_big',action="store_true",default=False)
	parser.add_argument('--overwrite',action="store_true",default=False)

	args = parser.parse_args()

	if args.step in ['download','everything']:
		print('cleaning %s'%(args.data_dir))
		harp_paper.common_paper.clean_empty(args.data_dir)

		print('Beginning download of <%.2fA structure into %s'%(args.cutoff,args.data_dir))
		harp_paper.download_rcsb(args.data_dir,args.cutoff)

	if args.step in ['calculate','everything']:
		print('Beginning HARP BMS calculations of <%.2fA structures from %s into %s'%(args.cutoff,args.data_dir,args.results_dir))
		harp_paper.calc_blobornot(args.data_dir,args.results_dir,args.cutoff,args.nworkers)

	if args.step in ['radial','everything']:
		print('Beginning radial calculations of <%.2fA structures from %s into %s'%(args.cutoff,args.data_dir,args.results_dir))
		harp_paper.calc_radial(args.data_dir,args.results_dir,args.cutoff,args.nworkers)
	
	if args.step in ['disthists','everything']:
		print('Beginning disthist calculations of <%.2fA structures from %s into %s'%(args.cutoff,args.data_dir,args.results_dir))
		harp_paper.calc_disthist(args.data_dir,args.results_dir,args.cutoff,args.nworkers)

	if args.step in ['metadata','everything']:
		print('Beginning .mmcif metadata extraction of <%.2fA structures from %s into %s'%(args.cutoff,args.data_dir,args.results_dir))
		harp_paper.extract_metadata(args.data_dir,args.results_dir,args.cutoff,args.overwrite)

	if args.step in ['progress','everything']:
		print('Checking progress of <%.2fA structures from %s into %s'%(args.cutoff,args.data_dir,args.results_dir))
		harp_paper.progress_check(args.data_dir,args.results_dir,args.cutoff)

	if args.step in ['check']:
		print('Checking status of databases in %s'%(args.data_dir))
		harp_paper.db_status(args.data_dir)

	if args.step in ['results','everything']:
		print('Collecting metadata and HARP calculation results of <%.2fA structures in %s'%(args.cutoff,args.results_dir))
		harp_paper.gather_results(args.data_dir,args.results_dir,args.cutoff)
		import shutil
		shutil.copy(os.path.join(args.results_dir,'all_results.hdf5'),'./all_results.hdf5')
	

	if args.step == 'remove_empty':
		print('cleaning %s'%(args.data_dir))
		harp_paper.common_paper.clean_empty(args.data_dir)
		print('cleaning %s'%(args.results_dir))
		harp_paper.common_paper.clean_empty(args.results_dir)

	print('========= Done =========')
