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

## Figures
scripts that generate figures
### Figures/rendered
.pdf and .png of the rendered figures




## Notes
Updating download on 12/27/2022

### 3IZO
3IZO it seems that the map in the emdb has been removed. manually remove entry from the pdb id list

### Complex size
NOTE ONLY USE FW. It's based off the deposited sequence and number of copies in the structure.
I've killed MW in common figures.




