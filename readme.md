# HARP Manuscript
This contains scripts to set up all of the HARP calculations, *etc.* presented in the HARP manuscript.

## Install the Requirements
The `harp` module will run a HARP calculation, but the make all of the figures, *etc.* in the manuscript, there are several more modules that are required (*e.g.*, `matplotlib`). You can install them by running:

``` bash
pip install -r requirements.txt
```

and, for a few figures you might also need:
* obabel
* dssp

which you should install using `apt` or `homebrew`... probably.

## Calculations
standalone scripts that don't really generate figures

### `residue_dispersion.py`
Calculate the variance of the atom positions in all residues in 8b0x (the ribosome has lots of every type of residue)

### `opt_weights_globalsigma_(pdbid).py`
These calculate optimal element weights (carbon fixed at 1) from the atoms in (pdbid). It uses a globally optimized sigma_ADF for the atoms.

## Figures
Scripts that generate figures. Final, rendered figures are stored in `/figures/rendered` as .pdf and .png format files. Generally you should run these from the top level directory as `python figures/(script_name_here).py`. Many of the EPres scripts require you to have `all_results.hdf5` and `all_radial.hdf5` present in the top directory.


### `hierarchy_of_blobs.py`
Displays a density model of 7a4m at 0: atomic resolution, 1: residue resolution, 2: local resolution (kmeans the COM of residues into 20 classes and predict the best class for each residue COM. Use that class assignment to calculate size of blob), or 3: global molecular resolution. You have to snap the figures using blobview -- so maybe update and do a MIP or someother projection.

### `view_scheres.py`
Finds a two conformation residue in 7a4m, visual it and the density in blobview. You must snap a picture in blobview.

### `theoretical_size.py`
Makes plots of theoretical atom profiles in an EM image by element type. Makes two -- one with B = 0 and another with a very low (cold) B factor from a Frauenfelder paper.

### `abbe_resolution_example.py`
makes a figure of two gaussians merging together


### `global_adf_offset.py`
scan through all sigma_adfs for several molecules, create models at each where all atoms are that size.

### `global_blob_sigma.py`
scan through all sigma_blobs for several molecules, create models at each where the M1 blob is that size.


### `global_adfweights.py`
scan through all sigma_adfs for several molecules, create models at each where all atoms are that size, and where the elements weighted by various schemes.

### `AA_cartoon.py`
make pictures of amino acids and then use it to describe the blobornot model

### `common_infer_full_2....py`
these are the hierarchical model functions. they don't make figures themselves.

### `common_figures.py`
these are functions that are used a lot in figure scripts

### `byresidue_posteriors.py`
generate several different plots of posteriors for each residue in 7a4m or 8b0x.

### `EPres_***.py`
These use `all_results.hdf5` to build hierarchical models


## Notes
Updating download on 12/27/2022

### 3IZO
3IZO it seems that the map in the emdb has been removed. manually remove entry from the pdb id list

### 7OJF
7OJF seems to be missing a map 

### Complex size
NOTE ONLY USE FW. It's based off the deposited sequence and number of copies in the structure.
I've killed MW in common figures.




