# HARP Manuscript
This contains scripts to set up all of the HARP calculations, *etc.* presented in the HARP manuscript.

## Run Calculations
The `run_paper.py` script allows you to work through all the steps required to calculate $\{P\}$ and extract all requird data for all the cryoEM structures in the PDB. Use the following flags

* `--data_dir /file/path/here` for where the PDB and EMDB files will be stored
* `--results_dir /file/path/here` for where the results files will be stored
* `--cutoff 8.0` sets the FSC resolution cutoff. All structures with resolution value below this cutoff will be downloaded and processed. Note this generates a list that is run once in order to freeze the calculation; the list is loaded every subsequent time.
* `--cutoff_date 2023-01-01` is the cutoff date for release to the PDB.

### Steps
Add one these arguments to execute that step (*e.g.*, `python run_paper.py download`)
* `download`: use FTP to download the ~12k mmCIF and MRC files. These stored compressed and loaded in dynamically to save space. ~1.7 Tb required. Code in `./harp_paper/download_rcsb.py`
* `calculate`: Runs HARP on each structure. Note, Running too many workers at the same time may required too much memory and cause crashes -- especially on viral structures. Code in `./harp_paper/calc_blobornot.py`
* `radial`: Calculates distance of each residue COM from the COM of the entire molecule. Code in `./harp_paper/calc_radial.py`
* `metadata`: Extract metadata used for plots from mmCIF files. Code in `./harp_paper/extract_metadata.py`
* `progress`: Check whether all the steps for each structure have been completed. Useful for finding incompletes. Code in `./harp_paper/progress_check.py`
* `check`:  Check how complete the DB is for all of the PDB_ID list files you have downloaded. Code in `./harp_paper/progress_check.py`
* `results`: Compile all the results into a single HDF5 file. Code in `./harp_paper/gather_results.py`
* `remove_empty`: Remove all files that have been touched, but do not have any data in them. Code in `./harp_paper/common_paper.py`
* `'disthists`: Calculates the distance between the closest residue for each residue of a molecule. Code in `./harp_paper/calc_M1_distances.py`


## Make Figures
The `figures/run_all_figures.py` script generates all figures. Once you have the `all_*.hdf5` files, then you can run `python figures/run_all_figures.py`. Note this should be run from the top-level directory, which is where the HDF5 files should reside. Rendered figures are stored in `/figures/rendered` as .pdf and .png format files. Generally you should run these from the top level directory as `python figures/(script_name_here).py`. Many of the EPres scripts require you to have `all_results.hdf5` and `all_radial.hdf5` present in the top directory.


### MS Figures

#### Fig. 2a
`figures/global_adfscan.py`

#### Fig. 2b
`figures/rayleigh_overlap.py`

#### Fig. 3
`figures/EPres_resolution_model.py`

#### Fig. 4a
`figures/EPres_year.py`

#### Fig. 4b
`figures/EPres_imgs.py`

#### Fig. 4c
`figures/EPres_imgs_lowres.py`

#### Fig. 4d
`figures/EPres_imgs_highres.py`

#### Fig. 4e
`figures/EPres_radial.py`

#### Fig. 5
`figures/EPres_residue_aa.py`

### SI Figures
#### Fig. S1
`figures/theoretical_size.py`

#### Fig. S2
`figures/global_blobsigma.py`

#### Fig. S3
`figures/inter_residue_distances_allpdb.py`

#### Fig. S5
`figures/EPres_camera.py`

#### Fig. S6
`figures/EPres_reconstructionsoftware.py`

#### Fig. S7
`figures/EPres_month.py`

#### Fig. S8
`figures/EPres_journal.py`

#### Fig. S9
`figures/EPres_dose.py`

#### Fig. S10
`figures/EPres_voltage.py`

#### Fig. S11
`figures/EPres_humidity.py`

#### Fig. S12
`figures/EPres_FW.py`

#### Fig. S13
`figures/EPres_residue_rg.py`

#### Fig. S14
`figures/EPres_residue_others.py`

#### Fig. S15
`figures/EPres_residue_others.py`

#### Fig. S16
`figures/EPres_residue_others.py`

#### Fig. S17
`figures/decarboxylation/decarb.py`


## Install the Requirements
The `harp` module will run a HARP calculation, but the make all of the figures, *etc.* in the manuscript, there are several more modules that are required (*e.g.*, `matplotlib`). You can install them by running:

``` bash
pip install -r requirements.txt
```


## Notes

### 3IZO
It seems that the map in the EMDB has been removed. Manually remove entry from the PDB ID list.

### 7OJF
Seems to be missing a map 





